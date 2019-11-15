from flask import Flask, g, request, Response, make_response
from keras import models
from keras import layers
from keras import optimizers, losses, metrics
from keras import preprocessing
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

from konlpy.tag import Twitter
import pickle
import time

# 태그 단어
PAD = "<PADDING>"   # 패딩
STA = "<START>"     # 시작
END = "<END>"       # 끝
OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

# 태그 인덱스
PAD_INDEX = 0
STA_INDEX = 1
END_INDEX = 2
OOV_INDEX = 3

# 데이터 타입
ENCODER_INPUT  = 0
DECODER_INPUT  = 1
DECODER_TARGET = 2

max_sequences = 30
RE_FILTER = re.compile("[.,!?\"':;~()]")

app = Flask(__name__)

# 모델 / 단어 가져오기
project_path = os.path.dirname(os.path.abspath(__file__))
encoder_model = load_model(project_path + "/data/encoder_model.h5")
decoder_model = load_model(project_path + "/data/decoder_model.h5")
with open(project_path + "/data/words.pickle", 'rb') as f:
   words = pickle.load(f)
words[:0] = [PAD, STA, END, OOV]

# 단어와 인덱스의 딕셔너리 생성
word_to_index = {word: index for index, word in enumerate(words)}
index_to_word = {index: word for index, word in enumerate(words)}

# 문장을 인덱스로 변환
def convert_text_to_index(sentences):
    sentences_index = []
    for sentence in sentences:
        sentence_index = []
        for word in sentence.split():
            if word_to_index.get(word) is not None:
                sentence_index.extend([word_to_index[word]])
            else:
                sentence_index.extend([word_to_index[OOV]])

        if len(sentence_index) > max_sequences:
            sentence_index = sentence_index[:max_sequences]

        # 패딩 인덱스
        sentence_index += (max_sequences - len(sentence_index)) * [word_to_index[PAD]]
        sentences_index.append(sentence_index)

    return np.asarray(sentences_index)

# 인덱스를 문장으로 변환
def convert_index_to_text(indexs):
    sentence = ''
    for index in indexs:
        if index == END_INDEX:
            break;

        if index == PAD_INDEX:
            continue;

        if index_to_word.get(index) is not None:
            sentence += index_to_word[index]
        else:
            sentence.extend([index_to_word[OOV_INDEX]])
        sentence += ' '

    return sentence

# 형태소분석 함수
def pos_tag(sentence):
    tagger = Twitter()
    sentence = re.sub(RE_FILTER, "", sentence)
    sentence = " ".join(tagger.morphs(sentence))
    return sentence

def predict(query):
    # encoder
    query = pos_tag(query)
    input_seq = convert_text_to_index([query])
    states = encoder_model.predict(input_seq)

    # decoder
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = STA_INDEX
    indexs = []
    while 1:
        decoder_outputs, state_h, state_c = decoder_model.predict([target_seq] + states)
        index = np.argmax(decoder_outputs[0, 0, :])
        indexs.append(index)

        # 종료 검사
        if index == END_INDEX or len(indexs) >= max_sequences:
            break

        # 목표 시퀀스를 바로 이전의 출력으로 설정
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = index

        # 디코더의 이전 상태를 다음 디코더 예측에 사용
        states = [state_h, state_c]

    # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs)
    return sentence

@app.route('/', methods=['POST'])
def index():
    query = request.values.get('query', 'default')
    print("Q: {0}".format(query))
    sentence = predict(query)
    print("A: {0}".format(sentence))
    return sentence