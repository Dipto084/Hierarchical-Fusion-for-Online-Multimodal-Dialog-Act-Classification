import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
import transformers
transformers.logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')
if torch.cuda.is_available():  
    print("Using GPU")   
    device = torch.device("cuda:5")
else:
    print("No GPU found")
    device = torch.device("cpu")

roberta = RobertaModel.from_pretrained("roberta-base")
roberta = roberta.to(device)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", padding = True, max_length = 400)

#either sum last 4 hidden layers or use last hidden state
def get_word_embeddings(sentence_batch, sum_hidden_states = False):
    #sum_hidden_states: if true, sums the token representations of last 4 hidden states. otherwise uses final hidden state only
    #sentence_batch: gets a list of strings, where each string is an utterance
    # with torch.no_grad():
    batch_encoded = tokenizer(sentence_batch, padding=True, truncation=True, return_tensors="pt").to(device)
    #contains input_ids and attention_mask
    attention_mask = batch_encoded.attention_mask
    out = roberta(**batch_encoded, output_hidden_states = sum_hidden_states)

    if sum_hidden_states == False: return out.last_hidden_state, attention_mask
                                    

    #sum last 4 hidden states
    embeddings = torch.stack(out.hidden_states, dim=0) #stack all layer outputs
    embeddings = embeddings.permute(1,2,0,3) #batch, tokens, layers, dim
    emb = embeddings[:,:,-4:,:].sum(dim = 2) #(batch, max tokens, dim)

    return emb, attention_mask
    #attention_mask: (batch size, max token)

class CalculateAttention(nn.Module):
    def __init__(self, config):
        super(CalculateAttention, self).__init__()

        self.ws1 = nn.Linear((config['hidden_dim']*2), (config['hidden_dim']*2))
        self.ws2 = nn.Linear((config['hidden_dim']*2), 1)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(config['dropout'])

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ws1.state_dict()['weight'])
        self.ws1.bias.data.fill_(0)
        nn.init.xavier_uniform(self.ws2.state_dict()['weight'])
        self.ws2.bias.data.fill_(0)

    def forward(self, outp, ctxt_mask = None):
        self_attention = F.relu(self.ws1(self.drop(outp)))
        self_attention = self.ws2(self.drop(self_attention)).squeeze()
        if ctxt_mask is not None:
            self_attention = self_attention + -10000*(ctxt_mask == 0).float()
        self_attention = self.drop(self.softmax(self_attention))
        sent_encoding = torch.sum(outp * self_attention.unsqueeze(-1), dim=1)

        return sent_encoding

#  Returns LSTM based sentence encodin, dim=1024, elements of vector in range [-1,1]
class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()

        self.sum_emb_representation = config["sum_emb_rep"]

        self.context_encoder_txt = nn.LSTM(config['embedding_dim'], config['hidden_dim'], config['num_layers'],
                                batch_first=True, bidirectional=config['bidirectional']) #, dropout=config['dropout']
        self.context_encoder_enc = nn.LSTM(config['hidden_dim'], config['hidden_dim'], config['num_layers'],
                                batch_first=True, bidirectional=config['bidirectional']) #, dropout=config['dropout']
        self.inner_pred = nn.Linear((config['hidden_dim']*4), config['hidden_dim']*4)
        self.drop = nn.Dropout(config['dropout'])
        self.txt_attention = CalculateAttention(config)
        self.enc_attention = CalculateAttention(config)

        self.init_weights()

    def init_weights(self):
        for name, param in self.context_encoder_enc.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.context_encoder_txt.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        nn.init.xavier_uniform(self.inner_pred.state_dict()['weight'])
        self.inner_pred.bias.data.fill_(0)

    def forward(self, text, enc_feats, hidden_ctxt = None):
        outp_emb, attention_mask = get_word_embeddings(text, self.sum_emb_representation)
        outp, _ = self.context_encoder_txt.forward(outp_emb, hidden_ctxt)
        txt_encoding = self.txt_attention(outp, attention_mask)
        outp_enc, _ = self.context_encoder_enc.forward(enc_feats, hidden_ctxt)
        enc_encoding = self.enc_attention(outp_enc)
        joint_encoding = torch.cat((txt_encoding, enc_encoding), dim = -1)
        joint_encoding = F.relu(self.inner_pred(self.drop(joint_encoding)))
        joint_encoding = joint_encoding.view(1,-1,joint_encoding.size(-1)) #(1, batchsize, 2048)
        return joint_encoding

class DiscourseEncoder(nn.Module):

    def __init__(self, config):
        super(DiscourseEncoder, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.discourse_encoder = nn.LSTM(config['hidden_dim']*4, config['hidden_dim']*4, config['num_layers'],
                              batch_first=True, bidirectional=False)
        self.pre_pred = nn.Linear((config['hidden_dim']*4), config['hidden_dim']*4)
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.discourse_encoder.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        nn.init.xavier_uniform(self.pre_pred.state_dict()['weight'])
        self.pre_pred.bias.data.fill_(0)

    def forward(self, sent_encoding, hidden_ctxt=None):
        inner_pred, hidden_op = self.discourse_encoder.forward(sent_encoding)
        inner_pred = inner_pred.squeeze()
        inner_pred = F.relu(self.pre_pred(self.drop(inner_pred)))
        
        return inner_pred

class Classifier(nn.Module):

    def __init__(self, config):
        super(Classifier, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.discourse_encoder = DiscourseEncoder(config)
        self.pre_pred = nn.Linear((config['hidden_dim']*4), config['hidden_dim']*2)
        self.pred = nn.Linear((config['hidden_dim']*2), config['out_dim'])
        self.drop = nn.Dropout(config['dropout'])
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.pre_pred.state_dict()['weight'])
        self.pre_pred.bias.data.fill_(0)
        nn.init.xavier_uniform(self.pred.state_dict()['weight'])
        self.pred.bias.data.fill_(0)

    def forward(self, text, enc_feats, hidden_ctxt = None):
        comb_pred = self.sentence_encoder(text, enc_feats)
        comb_pred = self.discourse_encoder(comb_pred)
        pre_pred = F.relu(self.pre_pred(self.drop(comb_pred)))
        pred = self.pred(self.drop(pre_pred))
        return pred