import torch
from transformers import WhisperFeatureExtractor, WhisperModel, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
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
import warnings
import csv
import h5py

warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"


device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
utterance = namedtuple('utterance', ['speaker_id', 'text', 'tag1', 'tag2', 'tag3', 'path'])

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base.en")
feat_model = WhisperModel.from_pretrained("openai/whisper-base.en").cuda(5)
processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
gen_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").cuda(5)
gen_model.config.forced_decoder_ids = None


class MRDADataset(Dataset):

    def __init__(self, batch_paths, transformation = None, target_sample_rate = 16000):
        self.batch_paths = batch_paths
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.transformation = transformation


    def __len__(self):
        return len(self.batch_paths)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal, sr = librosa.load(audio_sample_path, sr = self.target_sample_rate)
        
        if self.transformation != None:
            signal, mask = self.transformation(signal)
        
        return signal, mask

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            signal = librosa.resample(signal, orig_sr = sr, target_sr = self.target_sample_rate)
        return signal

    def _get_audio_sample_path(self, index):
        path = self.batch_paths[index]
        return path

class WhisperFeats(object):

    def __init__(self, sample_rate = 16000):
        self.sample_rate = sample_rate

    def __call__(self, signal):

        inputs = feature_extractor(signal, sampling_rate = self.sample_rate, return_attention_mask = True, return_tensors="pt")
        input_features = inputs.input_features
        attention_masks = inputs.attention_mask

        attention_len = torch.sum(attention_masks == 1).item()



        
        return input_features.squeeze(), attention_len

class WhisperIntegratedModel(nn.Module):

    def __init__(self, whisper_model_name, batch_size):
        super(WhisperIntegratedModel, self).__init__()
        self.whisper_model_name = whisper_model_name
        self.output_dim = whisper_models_dict[self.whisper_model_name]
        self.batch_size = batch_size

    def forward(self, audio, mask = None):
        
        decoder_input_ids = torch.tensor([[feat_model.config.decoder_start_token_id]] * audio.shape[0])
        
        if has_cuda:
            audio = audio.cuda(5)
            decoder_input_ids = decoder_input_ids.cuda(5)

        output_enc = feat_model(audio, attention_mask = mask, decoder_input_ids = decoder_input_ids).encoder_last_hidden_state
        
        print(output_enc.shape)
    
        return output_enc

        

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
        out.append(out_map[utterance.tag3])
        paths.append(utterance.path)
      
    return out, paths


def extract(data, fnames, batch_size, flag = 'train'):

    temp = list(zip(data, fnames))
    random.shuffle(temp)
    data, fnames = zip(*temp)
    data, fnames = list(data), list(fnames)

    start_time = time.time()
    
    for ind, doc in tqdm(enumerate(data)):
        model.train()
        encoder_representations = []
        encoder_masks = []
        y_true, audio_paths = get_context(doc)
        seq_len = len(y_true)

        if os.path.exists(os.path.join(base_path_enc, flag, fnames[ind][:-4] + '.h5')):
            print(fnames[ind])
            continue
        
        for batch_index in tqdm(range(0, seq_len, batch_size)):
 
            batch_paths = audio_paths[batch_index: min(batch_index + batch_size, seq_len)]

            train_set = MRDADataset(batch_paths, transformation = WhisperFeats())
            batch_audio = DataLoader(train_set, batch_size = batch_size, shuffle = False, num_workers = 4)

            audios, masks = next(iter(batch_audio))

            if has_cuda:
                audios = audios.cuda(5)

            output_enc = model(audios)

            output_enc = output_enc.cpu().detach().numpy()
            encoder_representations.append(output_enc)
            encoder_masks.extend(list(masks))

        print(fnames[ind])
        print(output_enc.shape)
        print(len(encoder_masks))

        with h5py.File(os.path.join(base_path_enc, flag, fnames[ind][:-4] + '.h5'), 'w') as hf:
            encoder_representations = np.vstack(encoder_representations)
            print(encoder_representations.shape)
            hf.create_dataset('encoder_representations', data = encoder_representations, compression = 'gzip', compression_opts=9)

        with h5py.File(os.path.join(base_path_enc, flag, fnames[ind][:-4] + '_mask.h5'), 'w') as hf:
            mask_array = np.zeros((len(encoder_masks), 1500), dtype=np.int32)
            for i, value in enumerate(encoder_masks):
                num_ones = value // 2 + 1
                mask_array[i, :num_ones] = 1
            print(mask_array.shape)
            hf.create_dataset('masks', data = mask_array, compression = 'gzip', compression_opts=9)


if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='SEED', default=0, type=int)
    parser.add_argument('--batch_size', help='Batch Size', default=50, type=int)

    args = parser.parse_args()
    has_cuda = torch.cuda.is_available()
    batch_size = args.batch_size
    seed = args.seed

    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
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

    # out_map = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
    
    reversed_map = {value : key for (key, value) in out_map.items()}

    whisper_models_dict = {"openai/whisper-tiny": 384,
                        "openai/whisper-tiny.en": 384,
                        "openai/whisper-base": 512,
                        "openai/whisper-base.en": 512,
                        "openai/whisper-small": 768,
                        "openai/whisper-small.en": 768,
                        "openai/whisper-medium": 1024,
                        "openai/whisper-medium.en": 1024,
                        "openai/whisper-large": 1280
                        }

    print("All labels: ", out_map)

    model = WhisperIntegratedModel(whisper_model_name = 'openai/whisper-base.en', batch_size = batch_size)

    if has_cuda:
        model = model.cuda(5)

    print("Model Created")

    base_path_enc = './whisper_encoder_feats_mrda'

    os.makedirs(base_path_enc, exist_ok =  True)
    os.makedirs(os.path.join(base_path_enc, 'train'), exist_ok =  True)
    os.makedirs(os.path.join(base_path_enc, 'test'), exist_ok =  True)
    os.makedirs(os.path.join(base_path_enc, 'val'), exist_ok =  True)

    try:
        for epoch in range(1):
            print("---------------------------Started Extraction--------------------------".format(epoch+1))
            extract(train_data, fnames_train, batch_size, 'train')
            extract(test_data, fnames_test, batch_size, 'test')
            extract(validate_data, fnames_val, batch_size, 'val')

    except KeyboardInterrupt:
        print ("----------------- INTERRUPTED -----------------")
