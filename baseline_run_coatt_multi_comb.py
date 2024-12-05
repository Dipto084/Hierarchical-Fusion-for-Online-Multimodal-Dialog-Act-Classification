from locale import currency
from logging import lastResort
import os
import h5py
import random
import torch
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from baseline_model_coatt_multi_comb import Classifier
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from copy import deepcopy
from collections import namedtuple
import pandas as pd
from sklearn.utils import shuffle

np.set_printoptions(threshold=np.inf)

import warnings
warnings.filterwarnings('ignore')
if torch.cuda.is_available():     
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

utterance = namedtuple('utterance', ['speaker_id', 'text', 'tag1', 'tag2', 'tag3'])

def process_file(dirname):
    print('baseline_run.process_file')
    data = []
    name = []
    fnames = os.listdir(dirname)
    for fname in fnames:
        f = open(dirname+fname, 'r')
        doc = []
        for l in f:
            speaker_id, text, t1, t2, t3, path = l.strip().split('|')
            doc.append(utterance(speaker_id, text.strip(), t1.strip(), t2.strip(), t3.strip()))
        data.append(doc)
        name.append(fname)
    return data, name


def get_data(fpath):
    print('baseline_run.get_data')
    return (process_file(fpath+'train/'),
            process_file(fpath+'val/'),
            process_file(fpath+'test/'))


def get_context(dialog):
    print('baseline_run.get_context')
    all_sent = []
    out = []
    last_spk = 'None'
    for index, utterance in enumerate(dialog):
        out.append(out_map[utterance.tag1])
        cur_speaker = utterance.speaker_id
        spk = '<same> ' if cur_speaker == last_spk else '<switch> '
        all_sent.append(spk + deepcopy(utterance.text[:200]))
        last_spk = cur_speaker

    assert len(out) == len(all_sent)
    return all_sent, out



def train(epoch, data, names, window_size, batch_size, log_dict):

    data, names = shuffle(data, names)

    start_time = time.time()
    classifier_loss = 0
    optimizer.zero_grad()
    global prev_best_macro
    for doc, name in tqdm(zip(data, names)):
        model.train()
        sent, y_true = get_context(doc) 
        seq_len = len(sent)

        if name in feats_dic.keys():
            feats_enc = feats_dic[name]
        
        else:
            with h5py.File(os.path.join(feats_path, 'train', name[:-4] + '.h5'), 'r') as hf:
                feats_enc = np.array(hf['encoder_representations'])

            feats_dic[name] = feats_enc

        for batch_index in range(0, len(sent), batch_size):
            sent_flat_txt = sent[max(0, batch_index-window_size) : min(batch_index + batch_size, seq_len)]
            
            sent_flat_enc = torch.Tensor(feats_enc[max(0, batch_index-window_size) : min(batch_index + batch_size, seq_len)]).to(device)

            out = torch.LongTensor(y_true[max(0, batch_index-window_size) : min(batch_index + batch_size, seq_len)]).to(device)
            output = model.forward(sent_flat_txt, sent_flat_enc)
            loss = criterion(output, out)
            classifier_loss += loss.item()
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    print("--Training--\nEpoch: ", epoch, "Discourse Act Classification Loss: ", classifier_loss,
          "Time Elapsed: ", time.time()-start_time)

    log_dict["epoch"].append(epoch)
    log_dict["loss"].append(classifier_loss)

    print ("-------------------Test start-----------------------")
    perf, cr1 = evaluate(validate_data, validate_names, window_size, batch_size, scores_dict)
    #if prev_best_macro < perf:
    prev_best_macro = perf
    print ("-------------------Test start-----------------------")
    _, cr2 = evaluate(test_data, test_names, window_size, batch_size, scores_dict, True)
    print ("-------------------Test end-----------------------")
    print ("Started saving model")
    torch.save(model.state_dict(), './model/mrda_coatt_multi_comb_5cls_batch_size_'+str(batch_size)+'_seed'+str(seed)+'.pt')
    print("Completed saving model")

    return cr1

def evaluate(data, names, window_size, batch_size, log_dict, is_test = False):
    print('baseline_run.evaluate')
    y_true, y_pred = [], []
    model.eval()

    for doc, name in tqdm(zip(data, names)):
        sent, out = get_context(doc)
        y_true += out
        seq_len = len(sent)

        if is_test:
            s = 'test'
        else:
            s = 'val'

        if name in feats_dic.keys():
            feats_enc = feats_dic[name]
        
        else:
            with h5py.File(os.path.join(feats_path, s, name[:-4] + '.h5'), 'r') as hf:
                feats_enc = np.array(hf['encoder_representations'])

            feats_dic[name] = feats_enc

        y_doc = []

        for batch_index in range(0, len(sent), batch_size):
            
            sent_flat_txt = sent[max(0, batch_index-window_size) : min(batch_index + batch_size, seq_len)]
            
            sent_flat_enc = torch.Tensor(feats_enc[max(0, batch_index-window_size) : min(batch_index + batch_size, seq_len)]).to(device)

            with torch.no_grad():
                output = model.forward(sent_flat_txt, sent_flat_enc)
            _, predict = torch.max(output, 1)
            start_index = 0 if batch_index == 0 else window_size
            y_pred += list(predict[start_index:].cpu().numpy() if has_cuda else predict.numpy())
            y_doc += list(predict[start_index:].cpu().numpy() if has_cuda else predict.numpy())
            
        if is_test:

            print("inference")

            current_directory = os.getcwd()
            final_directory = os.path.join(current_directory, f'prediction_out_self_multi_comb_5cls_{seed}/{name}')
            file1 = open(final_directory,"w")

            if len(doc) != len(y_doc):
                print("******STOP******")

            for utt, label in zip(doc, y_doc):
                file1.writelines("{}|{}|{}|{}|{}\n".format(utt.speaker_id, utt.text, utt.tag1, utt.tag2, reversed_map[label]))
            
            file1.close()

            print("inference complete")

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
        log_dict["test_micro_f1"][-1] = precision_recall_fscore_support(y_true, y_pred, average='micro')[1]
    
    return precision_recall_fscore_support(y_true, y_pred, average='macro')[2], classification_report(y_true, y_pred)


if __name__ == '__main__':
    print('baseline_run.__main__')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--drop', help='DROP', default=0.5, type=float)
    parser.add_argument('--learn_rate', help='LEARNING RATE', default=5e-4, type=float)
    parser.add_argument('--seed', help='SEED', default=0, type=int)
    parser.add_argument('--batch_size', help='Batch Size', default=100, type=int)
    parser.add_argument('--window_size', help='Sliding window size', default=10, type=int)
    parser.add_argument('--sum_embs', help='Sum last four hidden reps to get word emb', default=0, type=int)


    args = parser.parse_args()
    has_cuda = torch.cuda.is_available()
    drop = args.drop
    learn_rate = args.learn_rate
    seed = args.seed
    batch_size = args.batch_size
    window_size = args.window_size
    sum_emb_representation = args.sum_embs
    best_of_all_settings = []

    seed_list = [28, 420, 576]
    setting_list = [(210, 3)]
    for setting in setting_list:
        batch_size = setting[0]
        window_size = setting[1]


    (train_data, train_names), (validate_data, validate_names), (test_data, test_names) = get_data('./mrda_data_aligned/')
    print("Training Data: {0}, Validation Data: {1}, Test Data: {2}".format(len(train_data), len(validate_data), len(test_data)))
    # out_map = {'tc': 0, 'h': 1, 'nd': 2, 'qw': 3, 'df': 4, 't1': 5, 'na': 6, 'cs': 7, 'qh': 8, 'by': 9, 'g': 10,
    #             'ng': 11, 'no': 12, 'bs': 13, 'fe': 14, 'rt': 15, 'd': 16, 't3': 17, 'fa': 18, 'br': 19, 'qrr': 20,
    #             'arp': 21, 'qr': 22, 'r': 23, 'aap': 24, '2': 25, 'e': 26, 'cc': 27, 'ba': 28, '%': 29, 'b': 30,
    #             'fw': 31, 'j': 32, 'bk': 33, 'bsc': 34, 's': 35, 'qo': 36, 'co': 37, 'bh': 38, 'aa': 39, 'ft': 40,
    #             'qy': 41, 'fh': 42, 'm': 43, 'ar': 44, 'f': 45, 'bc': 46, 'bu': 47, 't': 48, 'am': 49, 'fg': 50, 'bd': 51}

    # out_map = {'g': 0, 'q': 1, 'ans': 2, 'o': 3, 's':4, 'ap': 5, 'c': 6, 'ag': 7, 'dag': 8, 'a': 9, 'b': 10, 'oth': 11}
    out_map = {'F':0, 'S': 1, 'B': 2, 'Q': 3, 'D': 4}

    reversed_map = {value : key for (key, value) in out_map.items()}

    feats_path = 'whisper_encoder_feats(1500x512)'
    feats_dic = {}

    for seed in seed_list:

        try:
            os.mkdir(f'./prediction_out_self_multi_comb_5cls_{seed}')
        except:
            pass

        print ("[HYPERPARAMS] dropout:{0}, sliding window size:{1}, learning rate:{2}, seed:{3}".format(drop, window_size, learn_rate, seed))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if has_cuda:
            torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        prev_best_macro = 0.
        model = Classifier({'num_layers': 1, 'hidden_dim': 128, 'bidirectional': True, 'embedding_dim': 768,
                            'audio_embedding_dim': 512, 'dropout': drop, 'out_dim': len(out_map), 
                            'sum_emb_rep': sum_emb_representation, 'window_size': window_size})
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(total_params)
        model.init_weights()
        criterion = nn.CrossEntropyLoss()
        print("Model Created")


        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params, lr=learn_rate, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)

        try:
            scores_dict = {"epoch": [], "loss": [], "val_macro_f1": [], "val_micro_f1": [], "test_macro_f1": [], "test_micro_f1": []}
            for epoch in range(20):
                print("---------------------------Started Training Epoch = {0}--------------------------".format(epoch+1))
                cr = train(epoch, train_data, train_names, window_size, batch_size, scores_dict)
                print(scores_dict)
                df = pd.DataFrame(scores_dict)
                df.to_csv(f'./log_self_multi_comb_5cls_{seed}.csv')
                try:
                    os.mkdir(f'./classification_reports_self_multi_comb_5cls_{seed}')
                except:
                    pass

                with open(f'./classification_reports_self_multi_comb_5cls_{seed}/cr_{epoch}.txt', mode='w') as file:
                    file.write(cr)

        except KeyboardInterrupt:
            print ("----------------- INTERRUPTED -----------------")
            print("best_of_all_settings ", best_of_all_settings)

        best_of_all_settings.append(prev_best_macro)
    
    print()
    print("best_of_all_settings ", best_of_all_settings)