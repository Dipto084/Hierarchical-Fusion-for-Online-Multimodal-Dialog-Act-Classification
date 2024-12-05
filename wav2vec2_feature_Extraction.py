import torch
from transformers import WhisperFeatureExtractor, WhisperModel, WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import os
from tqdm import tqdm
from collections import namedtuple
import random
import torch
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoProcessor
import warnings
import csv
import h5py

warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
utterance = namedtuple('utterance', ['speaker_id', 'text', 'tag1', 'tag2', 'tag3', 'path'])

model_name = "facebook/wav2vec2-large-xlsr-53"

#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
#feat_model = Wav2Vec2Model.from_pretrained(model_name).to(device)
feat_model = Wav2Vec2Model.from_pretrained(model_name).to(device)

class MRDADataset(Dataset):

    def __init__(self, batch_paths, transformation = None, target_sample_rate = 16000):
        self.batch_paths = batch_paths
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.batch_paths)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal, sr = librosa.load(audio_sample_path, sr = self.target_sample_rate)
        # signal = self._resample_if_necessary(signal, sr)

        if self.transformation != None:
            signal = self.transformation(signal)

        return signal
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            signal = librosa.resample(signal, orig_sr = sr, target_sr = self.target_sample_rate)
        return signal

    def _get_audio_sample_path(self, index):
        path = self.batch_paths[index]
        return path

class Wav2vec2Feats(object):

    def __init__(self, sample_rate = 16000):
        self.sample_rate = sample_rate

    def __call__(self, signal):

        inputs = feature_extractor(signal, 
                                    return_attention_mask = True,
                                    return_tensors="pt", 
                                    truncation = True,
                                    max_length = 11*self.sample_rate,
                                    padding = 'max_length',
                                    sampling_rate=self.sample_rate)
        input_features = inputs.input_values
        masks = inputs.attention_mask
        
        return input_features.squeeze(), masks.squeeze()


class W2V2IntegratedModel(nn.Module):

    def __init__(self, model_name, batch_size, dropout = 0.3, num_labels = 52):
        super(W2V2IntegratedModel, self).__init__()
        self.model_name = model_name
        self.output_dim = 512
        self.dropout = dropout
        self.num_labels = num_labels
        self.batch_size = batch_size

    def forward(self, audio, mask):


        
        with torch.no_grad():
            output_enc = feat_model(audio, mask, output_attentions = True)
        output_feats = output_enc.extract_features
        
        return output_feats


def process_file(dirname):
    data = []
    fnames = os.listdir(dirname)
    for fname in fnames:
        f = open(dirname+fname, 'r')
        doc = []
        for l in f:
            speaker_id, text, t1, t2, t3, path = l.strip().split('|')
            doc.append(utterance(speaker_id, text.split(), t1, t2, t3, path))

        data.append(doc)
    return fnames, data


def get_data(fpath):
    return (process_file(fpath+'train/'),
            process_file(fpath+'val/'),
            process_file(fpath+'test/'))


def get_context(dialog):
    paths = []
    out = []

    for index, utterance in enumerate(dialog):
        out.append(out_map[utterance.tag3.strip()])
        paths.append(utterance.path)
      
    return out, paths


def train(data, fnames, batch_size, flag):

    temp = list(zip(data, fnames))
    random.shuffle(temp)
    data, fnames = zip(*temp)
    data, fnames = list(data), list(fnames)

    start_time = time.time()
    
    for ind, doc in enumerate(data):
        if os.path.exists(os.path.join(base_path_enc, flag, fnames[ind][:-4] + '.h5')):
            print('path_exists')
            continue
        
        encoder_representations = []

        y_true, audio_paths = get_context(doc)
        seq_len = len(y_true)
       
        for batch_index in tqdm(range(0, seq_len, batch_size)):
 
            batch_paths = audio_paths[batch_index: min(batch_index + batch_size, seq_len)]

            train_set = MRDADataset(batch_paths, transformation = Wav2vec2Feats())
            batch_audio = DataLoader(train_set, batch_size = batch_size, shuffle = False, num_workers = 4)

            audios, masks = next(iter(batch_audio))

            if has_cuda:
                audios = audios.cuda(1)
                masks = masks.cuda(1)

            output_enc = model(audios, masks)

            output_enc = output_enc.cpu().detach().numpy()
  
            encoder_representations.append(output_enc)


        with h5py.File(os.path.join(base_path_enc, flag, fnames[ind][:-4] + '.h5'), 'w') as hf:
            encoder_representations = np.vstack(encoder_representations)
            print(f'{ind}.{fnames[ind]} has shape {encoder_representations.shape}')
            hf.create_dataset('encoder_representations', data = encoder_representations, compression = 'gzip', compression_opts=9)

if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', help='Batch Size', default=64, type=int)

    args = parser.parse_args()
    has_cuda = torch.cuda.is_available()
    batch_size = args.batch_size
    seed = 0

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    np.random.seed(seed)
    torch.manual_seed(seed)
    if has_cuda:
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    (fnames_train, train_data), (fnames_val, validate_data), (fnames_test, test_data) = get_data('/home/messal944084/Desktop/Dialogue-Act-Classification-main/mrda_data_whisper_aligned/')
    print("Training Data: {0}, Validation Data: {1}, Test Data: {2}".format(len(train_data), len(validate_data), len(test_data)))

    out_map = {'tc': 0, 'h': 1, 'nd': 2, 'qw': 3, 'df': 4, 't1': 5, 'na': 6, 'cs': 7, 'qh': 8, 'by': 9, 'g': 10,
               'ng': 11, 'no': 12, 'bs': 13, 'fe': 14, 'rt': 15, 'd': 16, 't3': 17, 'fa': 18, 'br': 19, 'qrr': 20,
               'arp': 21, 'qr': 22, 'r': 23, 'aap': 24, '2': 25, 'e': 26, 'cc': 27, 'ba': 28, '%': 29, 'b': 30,
               'fw': 31, 'j': 32, 'bk': 33, 'bsc': 34, 's': 35, 'qo': 36, 'co': 37, 'bh': 38, 'aa': 39, 'ft': 40,
               'qy': 41, 'fh': 42, 'm': 43, 'ar': 44, 'f': 45, 'bc': 46, 'bu': 47, 't': 48, 'am': 49, 'fg': 50, 'bd': 51}

    # out_map = {'s': 0, 'q': 1, 'ans': 2, 'g': 3, 'ap': 4, 'c': 5, 'ag': 6, 'dag': 7, 'o': 8, 'a': 9, 'b': 10,
    #            'oth': 11}

    # out_map = {'neutral': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'joy': 4, 'surprise': 5, 'disgust': 6}

    reversed_map = {value : key for (key, value) in out_map.items()}


    model = W2V2IntegratedModel(model_name = 'facebook/wav2vec2-large-xlsr-53', batch_size = batch_size)

    if has_cuda:
        model = model.cuda(1)

    print("Model Created")

    base_path_enc = './w2v2_encoder_feats'

    try:
        print("---------------------------Started Extraction--------------------------")
        train(train_data, fnames_train, batch_size, 'train')
        train(test_data, fnames_test, batch_size, 'test')
        train(validate_data, fnames_val, batch_size, 'val')

    except KeyboardInterrupt:
        print ("----------------- INTERRUPTED -----------------")
