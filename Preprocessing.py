import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import json, os
import matplotlib.pyplot as plt
from AE import AE
import torch
import argparse


train_logs = pd.read_csv('Data/train_logs.csv')
#train_scores = pd.read_csv('Data/train_scores.csv')
#test_logs = pd.read_csv('Data/test_logs.csv')

#print(train_logs.head())

def infer_AE(PATH):
    global device 

    txt_chg = np.load('Data/txt_chg_AE.npz')
    tensor_tc = torch.tensor(txt_chg['txt_chg'])
    tensor_tc = tensor_tc.type(torch.float).to('cpu')
    tc_dataset = TensorDataset(tensor_tc,tensor_tc) # create your datset
    tc_dataloader = DataLoader(tc_dataset,batch_size=256) # create your dataloader

    ae = AE()
    checkpoint = torch.load('models/AE_checkpoint.pth')
    ae.load_state_dict(checkpoint['model_state_dict'])
    ae.eval()
    ae.to(device)
    txt_chg_ae = []
    

    for step,(x,y) in enumerate(tqdm(tc_dataloader)):
        x = x.to(device)
        txt_chg_ae.append(ae.encoder(x))

    txt_chg_ae = torch.cat(txt_chg_ae,0)
    print(txt_chg_ae.size())
    torch.save(txt_chg_ae, 'Data/txt_chg_ae.pt')



def preprocess_logs(log):
    #enc = OneHotEncoder()
    log['mean_time'] = (log['down_time']+log['up_time'])/2

    # TEXT CHANGE
    infer_AE('models/AE_checkpoint.pth')

    # ACTIVITY
    print('Encoding activity...')
    if 'Act_onehot.npy' not in os.listdir('Data/'):
        act = list(map(lambda x:'Move' if 'Move' in x else x, log.activity))
        act_dict = {a:i for i,a in enumerate(set(act))}
        #print(act_dict)
        act_np = np.zeros((log.shape[0],len(act_dict)))
        for i in tqdm(range(len(act))):
            act_np[i,act_dict[act[i]]] = 1
        with open("Data/Act.json", "w") as outfile:
            json.dump(act_dict, outfile)
    

    # EVENT
    print('Encoding event...')
    if 'UpEvent_onehot.npy' not in os.listdir('Data/') or 'DownEvent_onehot.npy' not in os.listdir('Data/'):
        down = list(map(lambda x:'q' if len(x)==1 else x, log['down_event']))
        up = list(map(lambda x:'q' if len(x)==1 else x, log['up_event']))
        both = up.copy()
        both.extend(down)
        #print(Counter(event))
        neglected = list(Counter(both).keys())[-10:]
        both = list(Counter(both).keys())[:-10] # neglect last 10 events, too few, make robust
        
        event_dict = {e:i for i,e in enumerate(both)}
        event_dict['Others'] = len(event_dict)
        #print(event_dict)

        down = list(map(lambda x:'Others' if x in neglected or len(x)<1 else x, down))
        up = list(map(lambda x:'Others' if x in neglected or len(x)<1 else x, down))

        down_np = np.zeros((log.shape[0],len(event_dict)))
        up_np = np.zeros((log.shape[0],len(event_dict)))

        for i in tqdm(range(len(down))):
            down_np[i,event_dict[down[i]]] = 1
        for i in tqdm(range(len(up))):
            up_np[i,event_dict[up[i]]] = 1


        with open("Data/Event.json", "w") as outfile:
            json.dump(event_dict, outfile)

    # REST
    print('Encoding rest...')
    if 'rest.npy.npy' not in os.listdir('Data/'):
        rest = log[['id','event_id','mean_time','action_time','cursor_position','word_count']].to_numpy()

    with open('Data/x_train.npz', 'wb') as f:
        np.savez(f, act=act_np, up=up_np,down=down_np,rest=rest)
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Choose device")
    parser.add_argument('-n','--device', default='cpu')
    args = parser.parse_args()
    print(args)
    device = args.device
    preprocess_logs(train_logs)
