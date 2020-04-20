# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:52:26 2020

@author: insun
"""


import tensorflow_hub as hub
import tensorflow as tf

from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, Input

from nltk.corpus import stopwords


bert_model_L12 = "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1"
bert_model_un_L12 = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_model_un_L24 = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(bert_model_un_L12, trainable=True)

# text에 스페셜 토큰 추가 후 토큰화
def bert_tokenizer(texts, tokenizer, max_len=1024) : 
  all_tokens = []
  all_masks = []
  all_segments = []
  nltk_stopwords = stopwords.words('english')
  
  for text in texts :
    text = tokenizer.tokenize(text)
    text = [word for word in text if not word in nltk_stopwords] # 불용어 제거
    text = text[: max_len-2] # so that we can add cls and sep tokens
    input_seq = ["[CLS]"] + text + ["[SEP]"]
    pad_len = max_len - len(input_seq)

    tokens = tokenizer.convert_tokens_to_ids(input_seq)
    tokens += [0] * pad_len
    pad_masks = [1] * len(input_seq) + [0] * pad_len
    segment_ids = [0] * max_len

    all_tokens.append(tokens)
    all_masks.append(pad_masks)
    all_segments.append(segment_ids)

  return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

# BERT 모델 파인튜닝 부분
# bert_layer : bert임베딩 층
def build_model(bert_layer, max_len=1024) :
  input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
  input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
  segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

  pooled_output, seq_output = bert_layer([input_word_ids, input_mask, segment_ids])
  clf_output = seq_output[:, 0, :] #Tensor("strided_slice:0", shape=(None, 768), dtype=float32)

  out = Dense(1, activation='sigmoid')(clf_output)

  model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

  return model
