import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from allennlp.modules.elmo import Elmo, batch_to_ids

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import json
import random

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import *
#from torchsummary import summary
from torchaudio.transforms import *
import warnings
from torchinfo import summary


CUDA = torch.cuda.is_available()
if CUDA:
    print("GPU being used")
else:
    print("No gpu found")

class MRDADataset(Dataset):

    def __init__(self, batch_paths, transformation, target_sample_rate = 16000):
        self.batch_paths = batch_paths
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = 3 * self.target_sample_rate

    def __len__(self):
        return len(self.batch_paths)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _get_audio_sample_path(self, index):
        path = self.batch_paths[index]
        return path

class Restructure(object):
    """Compute spectrograms, delta and delta delta of spectrograms and stack them."""

    def __init__(self, sample_rate = 16000, delta = True, delta_delta = True):
        self.sample_rate = sample_rate
        self.delta = delta
        self.delta_delta = delta_delta

    def __call__(self, signal):

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.sample_rate,
            n_fft = 1024,
            #hop_length = 376,
            n_mels = 128
            )

        mel_specgram = mel_spectrogram(signal)

        delta_mel_spectrogram = torchaudio.transforms.ComputeDeltas()
        delta_delta_mel_spectrogram = torchaudio.transforms.ComputeDeltas()

        delta_mel_specgram = delta_mel_spectrogram(mel_specgram)
        delta_delta_mel_specgram = delta_delta_mel_spectrogram(delta_mel_specgram)

        signal_restructured = torch.stack([
            mel_specgram.squeeze(), 
            delta_mel_specgram.squeeze(), 
            delta_delta_mel_specgram.squeeze()
            ])

        return signal_restructured


options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

#options_file = "../data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
#weight_file = "../data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 1, dropout=0.0, requires_grad=False)
if CUDA:
    elmo = elmo.cuda(0)



def get_word_embeddings(sentence):
    character_ids = batch_to_ids(sentence)
    if CUDA:
        character_ids = character_ids.cuda(0)
    embedding = elmo(character_ids)
    outp_ctxt = embedding['elmo_representations'][0]
    ctxt_mask = embedding['mask']
    return outp_ctxt, ctxt_mask


class BiLSTM(nn.Module):

    def __init__(self, config, is_pos=False):
        super(BiLSTM, self).__init__()
        self.bidirectional = config['bidirectional']
        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.embedding_dim = config['embedding_dim']

        self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim, config['num_layers'],
                              batch_first=True, bidirectional=config['bidirectional']) #, dropout=config['dropout']

    def init_weights(self):
        for name, param in self.bilstm.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)

    def forward(self, emb, len_inp, hidden=None):
        len_inp = len_inp.cpu().numpy() if CUDA else len_inp.numpy()
        len_inp, idx_sort = np.sort(len_inp)[::-1], np.argsort(-len_inp)
        len_inp = len_inp.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda(0) if CUDA else torch.from_numpy(idx_sort)
        emb = emb.index_select(0, Variable(idx_sort))

        emb_packed = pack_padded_sequence(emb, len_inp, batch_first=True)
        outp, _ = self.bilstm(emb_packed, hidden)
        outp = pad_packed_sequence(outp, batch_first=True)[0]

        idx_unsort = torch.from_numpy(idx_unsort).cuda(0) if CUDA else torch.from_numpy(idx_unsort)
        outp = outp.index_select(0, Variable(idx_unsort))
        return outp

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        num_directions = 2 if self.bidirectional else 1
        return (Variable(weight.new(self.num_layers*num_directions, batch_size, self.hidden_dim).zero_()),
                Variable(weight.new(self.num_layers*num_directions, batch_size, self.hidden_dim).zero_()))


#  Returns LSTM based sentence encodin, dim=1024, elements of vector in range [-1,1]
class SentenceEncoder(nn.Module):

    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        self.context_encoder = BiLSTM(config)
        self.inner_pred = nn.Linear((config['hidden_dim']*2), config['hidden_dim']*2) # Prafulla 3
        self.ws1 = nn.Linear((config['hidden_dim']*2), (config['hidden_dim']*2))
        self.ws2 = nn.Linear((config['hidden_dim']*2), 1)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(config['dropout'])

    def init_weights(self):
        nn.init.xavier_uniform(self.ws1.state_dict()['weight'])
        self.ws1.bias.data.fill_(0)
        nn.init.xavier_uniform(self.ws2.state_dict()['weight'])
        self.ws2.bias.data.fill_(0)
        nn.init.xavier_uniform(self.inner_pred.state_dict()['weight'])
        self.inner_pred.bias.data.fill_(0)
        self.context_encoder.init_weights()

    def forward(self, outp_ctxt, ctxt_mask, length, hidden_ctxt=None):
        outp = self.context_encoder.forward(outp_ctxt, length, hidden_ctxt)

        self_attention = F.tanh(self.ws1(self.drop(outp)))
        self_attention = self.ws2(self.drop(self_attention)).squeeze()
        self_attention = self_attention + -10000*(ctxt_mask == 0).float()
        self_attention = self.drop(self.softmax(self_attention))
        sent_encoding = torch.sum(outp*self_attention.unsqueeze(-1), dim=1)

        return F.tanh(self.inner_pred(self.drop(sent_encoding)))


class DiscourseEncoder(nn.Module):

    def __init__(self, config):
        super(DiscourseEncoder, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.discourse_encoder = nn.LSTM(config['hidden_dim']*2, config['hidden_dim']*2, config['num_layers'],
                              batch_first=True, bidirectional=False)

    def init_weights(self):
        for name, param in self.discourse_encoder.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)

    def forward(self, sent_encoding, hidden_ctxt=None):
        inner_pred = self.drop(sent_encoding)
        inner_pred, hidden_op = self.discourse_encoder.forward(inner_pred)
        # print(inner_pred.size(), inner_pred[:,-1,:].size())
        return inner_pred[:, -1, :]  # Last hidden state



class TextModel(nn.Module):

    def __init__(self, config = {'history_length': 7, 'num_layers': 1, 'hidden_dim': 512, 'bidirectional': True, 'embedding_dim': 1024, 'dropout': 0.4, 'out_dim': 128}):
        super(TextModel, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.discourse_encoder = DiscourseEncoder(config)
        self.pre_pred = nn.Linear((config['hidden_dim']*2), config['hidden_dim']*2)
        self.pred = nn.Linear((config['hidden_dim']*2), config['out_dim'])
        self.drop = nn.Dropout(config['dropout'])
        self.init_weights()
        self.out_dim = config['out_dim']
        self.length = config['history_length']

    def init_weights(self):
        self.sentence_encoder.init_weights()
        self.discourse_encoder.init_weights()
        nn.init.xavier_uniform(self.pre_pred.state_dict()['weight'])
        self.pre_pred.bias.data.fill_(0)
        nn.init.xavier_uniform(self.pred.state_dict()['weight'])
        self.pred.bias.data.fill_(0)

    def forward(self, sentence, length, history_len=7, hidden_ctxt=None):
        outp_ctxt, ctxt_mask = get_word_embeddings(sentence)
        #print(outp_ctxt.shape)
        sent_encoding = self.sentence_encoder.forward(outp_ctxt, ctxt_mask, length, hidden_ctxt)
        #print(sent_encoding.shape)
        # modify size
        sent_encoding = sent_encoding.view(-1,history_len,sent_encoding.size(-1))
        #print(sent_encoding.shape)
        sent_encoding = self.discourse_encoder.forward(sent_encoding)
        #print(sent_encoding.shape)
        pre_pred = F.tanh(self.pre_pred(self.drop(sent_encoding)))
        #print(pre_pred.shape)
        return self.pred(self.drop(pre_pred))


class CNNLayerNorm(nn.Module):

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 

class ResidualCNN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)
        
class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

class AvgPooling(nn.Module):
    def __init__(self):
      super(AvgPooling, self).__init__()
      
    def forward(self, x):
        #x (batch, time, feature)
        x = x.transpose(1, 2).contiguous() # (batch, feature, time)
        x = nn.AvgPool1d(x.size()[2])(x)
        return x.transpose(1, 2).contiguous() # (batch, time, feature) 

class ReshapeClassifier(nn.Module):
    def __init__(self):
      super(ReshapeClassifier, self).__init__()
      
    def forward(self, x):
        x = x.squeeze(1)
        return x

hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 5,
    "rnn_dim": 512,
    "out_dim": 128,
    "n_feats": 128,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "batch_size": 512,
    "epochs": 10
}

class AudioModel(nn.Module):

    def __init__(self, n_cnn_layers = hparams["n_cnn_layers"], n_rnn_layers = hparams["n_rnn_layers"], rnn_dim = hparams["rnn_dim"], out_dim = hparams["out_dim"], n_feats = hparams["n_feats"], stride=2, dropout=hparams["dropout"]):
        super(AudioModel, self).__init__()
        self.out_dim = out_dim
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(3, 32, 3, stride = stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])

        
        self.processor = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.avg_pooling = AvgPooling()

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim, out_dim),
        )
        self.reshape_classifier_output = ReshapeClassifier()

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.processor(x)
        x = self.avg_pooling(x)
        x = self.classifier(x)
        x = self.reshape_classifier_output(x)
        return x

class MultiModalModel(nn.Module):
    def __init__(self, text_model, audio_model, dropout = 0.3, num_labels = 52):
        super(MultiModalModel, self).__init__()
        self.lexical = text_model
        self.acoustic = audio_model
        self.output_dim = self.lexical.out_dim + self.acoustic.out_dim
        self.num_labels = num_labels

        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.output_dim, self.num_labels)

    def forward(self, text, length, audio):
        lexical_output = self.lexical.forward(text, length)
        acoustic_output = self.acoustic(audio)
        merged_output = torch.cat((lexical_output,acoustic_output), dim=1)
        merged_output = self.dropout(self.relu(merged_output))
        return self.linear(merged_output)