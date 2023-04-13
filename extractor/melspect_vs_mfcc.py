# -*- coding: UTF-8 -*-
'''
@Project ：pythonProject 
@File    ：melspect_vs_mfcc.py
@Author  ：jiangym
@Date    ：2023/4/11 下午3:21 
'''

import librosa
from librosa.feature import melspectrogram

import torch
import torchaudio
import torchaudio.transforms as T



filepath = '/home/jym/mytemp/1-137-A-32.wav'
y, sr = librosa.load(filepath, sr=16000)

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)  # n_mfcc返回MFCC的数量

mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
