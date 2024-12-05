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
    device = torch.device("cuda:4")
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

class CalculateSelfAttention(nn.Module):
    def __init__(self, config):
        super(CalculateSelfAttention, self).__init__()

        self.ws1 = nn.Linear((config['hidden_dim']*4), (config['hidden_dim']*4))
        self.ws2 = nn.Linear((config['hidden_dim']*4), 1)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(config['dropout'])

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ws1.state_dict()['weight'])
        self.ws1.bias.data.fill_(0)
        nn.init.xavier_uniform(self.ws2.state_dict()['weight'])
        self.ws2.bias.data.fill_(0)

    def forward(self, outp, ctxt_mask = None):
        self_attention = F.tanh(self.ws1(self.drop(outp)))
        self_attention = self.ws2(self.drop(self_attention)).squeeze(2)
        if ctxt_mask is not None:
            self_attention = self_attention + -10000*(ctxt_mask == 0).float()
        self_attention = self.drop(self.softmax(self_attention))
        sent_encoding = torch.sum(outp * self_attention.unsqueeze(-1), dim=1)

        return sent_encoding

class CalculateCrossAttention(nn.Module):
    def __init__(self, config):
        super(CalculateCrossAttention, self).__init__()

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

    def forward(self, outp1, outp2, feat = 'enc', ctxt_mask = None):
        self_attention = F.tanh(self.ws1(self.drop(outp1)))
        self_attention = self.ws2(self.drop(self_attention)).squeeze(2)
        if feat == 'txt':
            self_attention = self_attention + -10000*(ctxt_mask == 0).float()
        self_attention = self.drop(self.softmax(self_attention))
        sent_encoding = torch.sum(outp2 * self_attention.unsqueeze(-1), dim=1)

        return sent_encoding

#  Returns LSTM based sentence encodin, dim=1024, elements of vector in range [-1,1]
class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()

        self.sum_emb_representation = config["sum_emb_rep"]

        self.context_encoder_txt = nn.LSTM(config['embedding_dim'], config['hidden_dim'] * 2, config['num_layers'],
                                batch_first=True, bidirectional=config['bidirectional']) #, dropout=config['dropout']
        self.context_encoder_enc = nn.LSTM(config['audio_embedding_dim'], config['hidden_dim'] * 2, config['num_layers'],
                                batch_first=True, bidirectional=config['bidirectional']) #, dropout=config['dropout']
        self.inner_pred = nn.Linear((config['hidden_dim']*8), config['hidden_dim']*8)
        self.drop = nn.Dropout(config['dropout'])
        self.txt_attention = CalculateSelfAttention(config)
        self.enc_attention = CalculateSelfAttention(config)

        self.init_weights()

    def init_weights(self):
        for name, param in self.context_encoder_enc.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.context_encoder_txt.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        nn.init.xavier_uniform(self.inner_pred.state_dict()['weight'])
        self.inner_pred.bias.data.fill_(0)

    def forward(self, text, enc_feats, attention_mask, hidden_ctxt = None):
        outp, _ = self.context_encoder_txt.forward(text, hidden_ctxt)
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
        self.context_size = config['window_size']
        self.drop = nn.Dropout(config['dropout'])
        self.discourse_encoder = nn.LSTM(config['hidden_dim']*8, config['hidden_dim']*8, config['num_layers'],
                              batch_first=True, bidirectional=False)
        self.pre_pred = nn.Linear((config['hidden_dim']*8), 1)
        self.init_weights()

    def init_weights(self):
        for name, param in self.discourse_encoder.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        nn.init.xavier_uniform(self.pre_pred.state_dict()['weight'])
        self.pre_pred.bias.data.fill_(0)

    def contextualize_input_text(self, text, context_size):
        text = text.squeeze(0)
        batch_size, embed_dim = text.shape
        zero_pad = torch.zeros(context_size - 1, embed_dim).to(device)
        padded_text = torch.cat((zero_pad, text), dim = 0)

        contextualized_text = []

        for i in range(batch_size):
            contextualized_text.append(padded_text[i: i+context_size])

        return torch.stack(contextualized_text).to(device)

    def forward(self, sent_encoding, hidden_ctxt=None):
        sent_encoding, hidden_op = self.discourse_encoder.forward(sent_encoding)
        sent_encoding = self.contextualize_input_text(sent_encoding, self.context_size)
        inner_pred, hidden_op = self.discourse_encoder.forward(sent_encoding)
        inner_pred = inner_pred.permute(1, 0, 2)
        activations = [] 
        for i in range(self.context_size):
            activations.append(self.pre_pred(inner_pred[i]))
        # activations: [num sent, batch size, 1]
        activation = torch.stack(activations)

        attention = nn.Softmax(dim=0)(activation)
        inner_pred = torch.sum((attention * inner_pred),dim=0)
        return inner_pred

class LexicalModel(nn.Module):
    def __init__(self, config, conv_channels=256, kernel_size=5):
        super(LexicalModel, self).__init__()
        self.context_size = config['window_size']
        self.embedding_dim = config['embedding_dim']
        self.conv_channels = conv_channels
        self.output_dim = config['hidden_dim'] 
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(self.embedding_dim, self.conv_channels, self.kernel_size, padding = 'same')
        self.relu = nn.ReLU()
        self.dropout=nn.Dropout(p = config['dropout'])
        self.lstm = nn.LSTM(
            input_size = self.conv_channels,
            hidden_size = self.output_dim,
            batch_first = False) 

        self.attention_fc = nn.Linear(self.output_dim, 1)
        self.softmax = nn.Softmax(dim = 0)

    def contextualize_input_text(self, text, context_size):

        batch_size, seq_len, embed_dim = text.shape
        zero_pad = torch.zeros(context_size - 1, seq_len, embed_dim).to(device)
        padded_text = torch.cat((zero_pad, text), dim = 0)

        contextualized_text = []

        for i in range(batch_size):
            contextualized_text.append(padded_text[i: i+context_size])

        return torch.stack(contextualized_text).to(device)
        
    def forward(self, text):
        # 1. get embdding vectors
        # embedded: [num sent, batch size, emb dim, sent len]
        embedded = self.contextualize_input_text(text, self.context_size).permute(1,0,3,2)      
        len_ctxt, bs, dim, seq_len = embedded.shape
        # 2. convolution over each sentence
        conv_output = self.conv(embedded.reshape(-1, dim, seq_len))
        # conv_output: [num sent, batch size, num channels, sent len]
        conv_output = conv_output.reshape(len_ctxt, bs, -1, seq_len)
        
        # 3. MaxPool the whole sentence into a single vector of dim num channels
        # max_output: [num sent, batch size, num_channels]
        max_output,_ = torch.max(conv_output,dim = 3)  #max over sentence length
        max_output = self.dropout(self.relu(max_output))
        # 4. LSTM the 3 sentences to determine attention 
        # lstm_output: [num sent, batch size, output dim]
        lstm_output, _ = self.lstm(max_output)
        
        # 5. Activation units
        activation = self.attention_fc(lstm_output)
        # activation: [num sent, batch size, 1]

        # 6. Compute attention
        attention = self.softmax(activation)

        # 7. Sum the resulting vectors weighted by attention
        ret = torch.sum((attention * lstm_output), dim=0)
        return ret


class AcousticModel(nn.Module):
    def __init__(self, config, conv_channels=256, kernel_size=5):
        super(AcousticModel, self).__init__()
        self.context_size = config['window_size']
        self.embedding_dim = config['audio_embedding_dim']
        self.conv_channels = conv_channels
        self.output_dim = config['hidden_dim'] 
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(self.embedding_dim, self.conv_channels, self.kernel_size, padding = 'same')
        self.relu = nn.ReLU()
        self.dropout=nn.Dropout(p = config['dropout'])
        self.lstm = nn.LSTM(
            input_size = self.conv_channels,
            hidden_size = self.output_dim,
            batch_first = False) 

        self.attention_fc = nn.Linear(self.output_dim, 1)
        self.softmax = nn.Softmax(dim = 0)

    def contextualize_input_text(self, text, context_size):

        batch_size, seq_len, embed_dim = text.shape
        zero_pad = torch.zeros(context_size - 1, seq_len, embed_dim).to(device)
        padded_text = torch.cat((zero_pad, text), dim = 0)

        contextualized_text = []

        for i in range(batch_size):
            contextualized_text.append(padded_text[i: i+context_size])

        return torch.stack(contextualized_text).to(device)
        
    def forward(self, text):
        # 1. get embdding vectors
        # embedded: [num sent, batch size, emb dim, sent len]
        embedded = self.contextualize_input_text(text, self.context_size).permute(1,0,3,2)      
        len_ctxt, bs, dim, seq_len = embedded.shape
        # 2. convolution over each sentence
        conv_output = self.conv(embedded.reshape(-1, dim, seq_len))
        # conv_output: [num sent, batch size, num channels, sent len]
        conv_output = conv_output.reshape(len_ctxt, bs, -1, seq_len)
        
        # 3. MaxPool the whole sentence into a single vector of dim num channels
        # max_output: [num sent, batch size, num_channels]
        max_output,_ = torch.max(conv_output,dim = 3)  #max over sentence length
        max_output = self.dropout(self.relu(max_output))
        # 4. LSTM the 3 sentences to determine attention 
        # lstm_output: [num sent, batch size, output dim]
        lstm_output, _ = self.lstm(max_output)
        
        # 5. Activation units
        activation = self.attention_fc(lstm_output)
        # activation: [num sent, batch size, 1]

        # 6. Compute attention
        attention = self.softmax(activation)

        # 7. Sum the resulting vectors weighted by attention
        ret = torch.sum((attention * lstm_output), dim=0)
        return ret

class Classifier(nn.Module):

    def __init__(self, config):
        super(Classifier, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.discourse_encoder = DiscourseEncoder(config)
        self.lexical_encoder = LexicalModel(config)
        self.acoustic_encoder = AcousticModel(config)
        self.pre_pred = nn.Linear((config['hidden_dim']*8), config['hidden_dim']*2)
        self.pred = nn.Linear((config['hidden_dim']*4), config['out_dim'])
        self.drop = nn.Dropout(config['dropout'])
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.pre_pred.state_dict()['weight'])
        self.pre_pred.bias.data.fill_(0)
        nn.init.xavier_uniform(self.pred.state_dict()['weight'])
        self.pred.bias.data.fill_(0)

    def forward(self, text, enc_feats, hidden_ctxt = None):
        text, attention_mask = get_word_embeddings(text)
        lexical_output = self.lexical_encoder(text)
        acoustic_output = self.acoustic_encoder(enc_feats)
        comb_pred = self.sentence_encoder(text, enc_feats, attention_mask)
        comb_pred = comb_pred.squeeze(0)
        #comb_pred = self.discourse_encoder(comb_pred)
        #print(comb_pred.shape)
        pre_pred = F.relu(self.pre_pred(self.drop(comb_pred)))
        pre_pred = torch.cat((pre_pred, lexical_output, acoustic_output), dim = -1)
        pred = self.pred(self.drop(pre_pred))
        return pred
