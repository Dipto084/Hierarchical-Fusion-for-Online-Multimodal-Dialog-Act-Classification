import torch
from transformers import WhisperFeatureExtractor, WhisperModel
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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
utterance = namedtuple('utterance', ['speaker_id', 'text', 'tag1', 'tag2', 'tag3', 'path'])

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base.en")
feat_model = WhisperModel.from_pretrained("openai/whisper-base.en").cuda(1)

class MRDADataset(Dataset):

    def __init__(self, batch_paths, transformation, target_sample_rate = 16000):
        self.batch_paths = batch_paths
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.transformation = transformation


    def __len__(self):
        return len(self.batch_paths)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal, sr = librosa.load(audio_sample_path, sr = self.target_sample_rate)
        #signal = self._resample_if_necessary(signal, sr)
        signal = self.transformation(signal)
        
        return signal

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

        inputs = feature_extractor(signal, sampling_rate = self.sample_rate, return_tensors="pt")
        input_features = inputs.input_features
        
        return input_features.squeeze()

class WhisperIntegratedModel(nn.Module):

    def __init__(self, whisper_model_name, batch_size, dropout = 0.3, num_labels = 52):
        super(WhisperIntegratedModel, self).__init__()
        self.whisper_model_name = whisper_model_name
        self.output_dim = whisper_models_dict[self.whisper_model_name]
        self.dropout = dropout
        self.num_labels = num_labels
        self.batch_size = batch_size

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout)
        self.linear = nn.Linear(self.output_dim, self.num_labels)
        self.linear2 = nn.Linear(63, self.num_labels)
        self.lazylinear = nn.LazyLinear(self.num_labels)
        self.conv1 = nn.Conv1d(in_channels = 1500, out_channels = 512, kernel_size = 3, stride=2)
        self.conv2 = nn.Conv1d(in_channels = 512, out_channels = 64, kernel_size = 3, stride=2)
        self.conv3 = nn.Conv1d(in_channels = 64, out_channels = 1, kernel_size = 3, stride=2)
        self.maxpool = nn.MaxPool1d(3, stride = 2)

    def forward(self, audio):
        decoder_input_ids = torch.tensor([[feat_model.config.decoder_start_token_id]] * audio.shape[0])
        
        if has_cuda:
            audio = audio.cuda(1)
            decoder_input_ids = decoder_input_ids.cuda(1)

        output = feat_model(audio, decoder_input_ids = decoder_input_ids).encoder_last_hidden_state

        
        output_conv = self.conv1(output)
        output_pool = self.maxpool(output_conv)
        output_activation = self.relu(output_conv)

        output_conv = self.conv2(output_activation)
        output_pool = self.maxpool(output_conv)
        output_activation = self.relu(output_conv)

        output = self.dropout(output_activation)

        output_conv = self.conv3(output)
        output_pool = self.maxpool(output_conv)
        output_activation = self.relu(output_conv)

        output = self.dropout(output_activation)

        output = output.squeeze()
        # processed_audio = self.dropout(output)
        # processed_audio = self.linear(processed_audio)

        processed_audio = self.linear2(output)

        return processed_audio
    
    def init_weights(self):
        nn.init.xavier_uniform(self.linear.state_dict()['weight'])
        self.linear.bias.data.fill_(0)
        

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


def train(epoch, data, batch_size, log_dict):

    random.shuffle(data)
    start_time = time.time()
    classifier_loss = 0
    optimizer.zero_grad()
    global prev_best_macro
    for ind, doc in tqdm(enumerate(data)):
        model.train()
        y_true, audio_paths = get_context(doc)
        seq_len = len(y_true)
        for batch_index in tqdm(range(0, seq_len, batch_size)):
 
            out = torch.LongTensor(y_true[batch_index: min(batch_index + batch_size, seq_len)])
            batch_paths = audio_paths[batch_index: min(batch_index + batch_size, seq_len)]
            
            if has_cuda:
                out = out.cuda(1)

            train_set = MRDADataset(batch_paths, transformation = WhisperFeats())
            batch_audio = DataLoader(train_set, batch_size = batch_size, shuffle = False, num_workers = 4)

            audios = next(iter(batch_audio))


            if has_cuda:
                audios = audios.cuda(1)

            output = model(audios)
            
            if len(output.shape) != 2:
                output = output.unsqueeze(0)

            output = F.log_softmax(output, dim = 1)

            #print(output.shape, out.shape)
            loss = criterion(output, out)  # last sentence is eod

            classifier_loss += loss.item()
            loss.backward()

        print(classifier_loss)

        optimizer.step()
        optimizer.zero_grad()

    print("--Training--\nEpoch: ", epoch, "Discourse Act Classification Loss: ", classifier_loss,
          "Time Elapsed: ", time.time()-start_time)

    log_dict["epoch"].append(epoch)
    log_dict["loss"].append(classifier_loss)

    perf, cr1 = evaluate(validate_data, scores_dict, False, batch_size)
    if prev_best_macro < perf:
        prev_best_macro = perf
        print ("-------------------Test start-----------------------")
        _, cr2 = evaluate(test_data, scores_dict, True, batch_size, fnames = fnames_test)
        print ("-------------------Test end-----------------------")
        print ("Started saving model")
        torch.save(model.state_dict(), './model/baseline_model_whisper.pt')
        print("Completed saving model")

    return cr1


def evaluate(data, log_dict, is_test = False, batch_size=100, fnames = None):

    y_true, y_pred = [], []
    model.eval()

    for i, doc in tqdm(enumerate(data)):
        out, audio_paths = get_context(doc)
        y_true += out
        seq_len = len(out)

        y_doc = []
        for batch_index in tqdm(range(0, seq_len, batch_size)):

            batch_paths = audio_paths[batch_index: min(batch_index + batch_size, seq_len)]

            valid_set = MRDADataset(batch_paths, transformation = WhisperFeats())
            batch_audio = DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 4)

            try:
                audios = next(iter(batch_audio))
            except:
                continue

            if has_cuda:
                audios = audios.cuda(1)

            with torch.no_grad():
                output = model(audios)

            if len(output.shape) != 2:
                output = output.unsqueeze(0)

            _, predict = torch.max(output, 1)
            y_pred += list(predict.cpu().numpy() if has_cuda else predict.numpy())
            y_doc += list(predict.cpu().numpy() if has_cuda else predict.numpy())
        
        #for exp

        if is_test:
            current_directory = os.getcwd()
            final_directory = os.path.join(current_directory, r'prediction_out/{}'.format(fnames[i]))
            file1 = open(final_directory,"w")
            if len(doc) != len(y_doc):
                print("******STOP*****")

            for utt, label in zip(doc, y_doc):
                file1.writelines("{}|{}|{}|{}|{}\n".format(utt.speaker_id, " ".join(utt.text), utt.tag1, utt.tag2, reversed_map[label]))
        
            file1.close()


    print("MACRO: ", precision_recall_fscore_support(y_true, y_pred, average='macro'))
    print("MICRO: ", precision_recall_fscore_support(y_true, y_pred, average='micro'))
    
    if not is_test:
        log_dict["val_macro_f1"].append(precision_recall_fscore_support(y_true, y_pred, average='macro')[2])
        log_dict["val_micro_f1"].append(precision_recall_fscore_support(y_true, y_pred, average='micro')[2])
        log_dict["test_macro_f1"].append('-')
        log_dict["test_micro_f1"].append('-')
    
    if is_test:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Classification Report \n", classification_report(y_true, y_pred))
        print("Confusion Matrix \n", confusion_matrix(y_true, y_pred))
        log_dict["test_macro_f1"][-1] = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
        log_dict["test_micro_f1"][-1] = precision_recall_fscore_support(y_true, y_pred, average='micro')[2]
    return precision_recall_fscore_support(y_true, y_pred, average='macro')[2], classification_report(y_true, y_pred)




if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--drop', help='DROP', default=0.3, type=float)
    parser.add_argument('--learn_rate', help='LEARNING RATE', default=1e-3, type=float)
    parser.add_argument('--seed', help='SEED', default=0, type=int)
    parser.add_argument('--history', help='# Historical utterances to consider', default=7, type=int)
    parser.add_argument('--batch_size', help='Batch Size', default=27, type=int)

    args = parser.parse_args()
    has_cuda = torch.cuda.is_available()
    drop = args.drop
    learn_rate = args.learn_rate
    seed = args.seed
    history_len = args.history
    batch_size = args.batch_size

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print ("[HYPERPARAMS] dropout: ", drop, "learning rate: ", learn_rate, "seed: ", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if has_cuda:
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    (fnames_train, train_data), (fnames_val, validate_data), (fnames_test, test_data) = get_data('/home/messal944084/Desktop/Dialogue-Act-Classification-main/mrda_data_aligned/')
    print("Training Data: {0}, Validation Data: {1}, Test Data: {2}".format(len(train_data), len(validate_data), len(test_data)))

    out_map = {'tc': 0, 'h': 1, 'nd': 2, 'qw': 3, 'df': 4, 't1': 5, 'na': 6, 'cs': 7, 'qh': 8, 'by': 9, 'g': 10,
               'ng': 11, 'no': 12, 'bs': 13, 'fe': 14, 'rt': 15, 'd': 16, 't3': 17, 'fa': 18, 'br': 19, 'qrr': 20,
               'arp': 21, 'qr': 22, 'r': 23, 'aap': 24, '2': 25, 'e': 26, 'cc': 27, 'ba': 28, '%': 29, 'b': 30,
               'fw': 31, 'j': 32, 'bk': 33, 'bsc': 34, 's': 35, 'qo': 36, 'co': 37, 'bh': 38, 'aa': 39, 'ft': 40,
               'qy': 41, 'fh': 42, 'm': 43, 'ar': 44, 'f': 45, 'bc': 46, 'bu': 47, 't': 48, 'am': 49, 'fg': 50, 'bd': 51}

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
    prev_best_macro = 0.

    model = WhisperIntegratedModel(whisper_model_name = 'openai/whisper-base.en', batch_size = batch_size)
    
    if has_cuda:
        model = model.cuda(1)

    #model.init_weights()
    #model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
    print("Model Created")

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=learn_rate, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)

    try:
        scores_dict = {"epoch": [], "loss": [], "val_macro_f1": [], "val_micro_f1": [], "test_macro_f1": [], "test_micro_f1": []}
        for epoch in range(20):
            print("---------------------------Started Training Epoch = {0}--------------------------".format(epoch+1))
            cr = train(epoch, train_data, batch_size, scores_dict)
            print(scores_dict)
            df = pd.DataFrame(scores_dict)
            df.to_csv('./log_whisper_enc.csv')
            try:
                os.mkdir('./classification_reports')
            except:
                pass
            try:
                df_cr = pd.DataFrame(cr)
                df.to_csv(f'./classification_reports/cr_{epoch}.csv')
            except:
                pass

    except KeyboardInterrupt:
        print ("----------------- INTERRUPTED -----------------")