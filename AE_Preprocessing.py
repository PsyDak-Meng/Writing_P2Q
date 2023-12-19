import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import json, os
import tensorflow as tf



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

def preprocess(log):
  # TEXT CHANGES
      print('Encode text_changes...')
      text_change = tf.keras.utils.pad_sequences(
          list(map(lambda x:parse_text_change(x),tqdm(log.text_change,desc='Encode text change:'))),
          maxlen=117,
          dtype='float16',
          padding='post',
          truncating='post',
          value=0.0
      )
      text_change = np.array(text_change)
      print(text_change.shape)

      if 'txt_chg_AE.npz' not in os.listdir('Data/'):
        with open('Data/txt_chg_AE.npz','wb') as f:
          np.savez(f, txt_chg = text_change)

if __name__=='__main__':
    train_logs = pd.read_csv('Data/train_logs.csv')
    #train_scores = pd.read_csv('Data/train_scores.csv')
    #test_logs = pd.read_csv('Data/test_logs.csv')

    preprocess(train_logs)
