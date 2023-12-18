import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
from collections import Counter
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import json, os
import tensorflow as tf
import matplotlib.pyplot as plt


train_logs = pd.read_csv('Data/train_logs.csv')
train_scores = pd.read_csv('Data/train_scores.csv')
test_logs = pd.read_csv('Data/test_logs.csv')

print(train_logs.head())

def parse_text_change(text):
    text_embed = text.split(' ') # embed by length
    for i in range(len(text_embed)):
        if 'q' in text_embed[i]:
            text_embed[i] = len(text_embed[i])
        elif text_embed[i] == 'NoChange':
            text_embed[i] = 0
        else:
            text_embed[i] = 0.5
    return text_embed

def preprocess_logs(log):
    #enc = OneHotEncoder()
    log['mean_time'] = (log['down_time']+['log.up_time'])/2

    # ACTIVITY
    print('Encoding activity...')
    if 'Act_onehot.npy' not in os.listdir('Data/'):
        act = list(map(lambda x:'Move' if 'Move' in x else x, log.activity))
        act_dict = {a:i for i,a in enumerate(set(act))}
        #print(act_dict)
        act_np = np.zeros((log.shape[0],len(act_dict)))
        for i in tqdm(range(len(act))):
            act_np[i,act_dict[act[i]]] = 1
        with open('Data/Act_onehot.npy', 'wb') as f:
            np.save(f, act_np)
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

        with open('Data/DownEvent_onehot.npy', 'wb') as f:
            np.save(f, down_np)
        with open('Data/UpEvent_onehot.npy', 'wb') as f:
            np.save(f, up_np)
        with open("Data/Event.json", "w") as outfile:
            json.dump(event_dict, outfile)


    # TEXT CHANGES
    print('Encode text_changes...')
    if 'TextChange_LengthCoded.npy' not in os.listdir('Data/'):
        text_change = tf.keras.utils.pad_sequences(
            list(map(lambda x:parse_text_change(x),tqdm(log.text_change,desc='Encode text change:'))),
            maxlen=250,
            dtype='int32',
            padding='post',
            truncating='post',
            value=0.0
        )
        text_change = np.array(text_change)
        print(text_change.shape)
        with open('Data/TextChange_LengthCoded.npy', 'wb') as f:
            np.save(f, text_change)

    # REST
    print('Encoding rest...')
    if 'rest.npy.npy' not in os.listdir('Data/'):
        rest = log[['id','event_id','mean_time','action_time','cursor_position','word_count']].to_numpy()
        with open('Data/rest.npy', 'wb') as f:
            np.save(f, rest)
    
if __name__ =='__main__':
    preprocess_logs(train_logs)
