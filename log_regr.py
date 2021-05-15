#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import matplotlib
import seaborn as sns

font1 = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font1)

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks, stft

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

get_ipython().run_line_magic('matplotlib', 'inline')

data_dir = './data/npy/augment'
fs = 100.0
data = []
for f in os.listdir(data_dir):
    data.append(np.load(os.path.join(data_dir, f)))
    
walk1 = np.load('./data/npy/walk1.npy')
walk2 = np.load('./data/npy/walk2.npy')
walk3 = np.load('./data/npy/walk_regr.npy')
walk_all = np.vstack((walk1,walk2))

train = walk_all[:270000,:]
test = walk_all[270000:,:]

conv_width = 1
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def get_scalogram_coefs(x, features, fs):
    all_coefs = []
    for feat in features:
        scales = np.arange(2, fs*2)
        coefs, freqs = pywt.cwt(x[:,feat], scales, 'cmor-1-1')

        coefs = np.abs(coefs)

        if all_coefs == []:
            all_coefs = coefs[0:100,:]

        else:
            all_coefs = np.vstack((all_coefs,coefs[0:100,:]))


        plt.figure(figsize=(15,8))
        plt.plot(x[:,feat])
        plt.xlabel('Samples')
        plt.ylabel('Normalized Amplitude')

        plt.figure(figsize=(15,8))
        plt.imshow(coefs, extent=[0, x.shape[0], 150, 2], cmap='jet', aspect='auto',
                    vmax=coefs.max(), vmin=-coefs.min())
        plt.title('CWT Scaleogram')
        plt.xlabel('Samples')
        plt.ylabel('Period (samples)')
        plt.show()
        
    return all_coefs

X_train = train[:,:9]
Y_train = train[:,9].astype(int)
X_test = test[:,:9]
Y_test = test[:,9].astype(int)

all_coefs_train = get_scalogram_coefs(X_train, [1,3,7], 100)
all_coefs_test = get_scalogram_coefs(X_test, [1,3,7], 100)
coefs_train_scaled = preprocessing.scale(all_coefs_train.T)
coefs_test_scaled = preprocessing.scale(all_coefs_test.T)

scaler = preprocessing.StandardScaler()
scaler.fit(coefs_train_scaled)
x_train = scaler.transform(coefs_train_scaled)
x_test = scaler.transform(coefs_test_scaled)
clf = LogisticRegression(random_state=0, verbose=1, max_iter=100).fit(x_train, Y_train)

test_pred = moving_average(clf.predict(x_test),conv_width)
plt.figure(figsize=(15,12))
plt.plot(test_pred, alpha=0.8, label='Predictions')
plt.plot(Y_test[conv_width-1:], linewidth=7, label='Labels')
plt.title('Logistic Regression Predictions')
plt.xlabel('Samples')
plt.ylabel('Digitized Slope (Degrees)')
plt.legend()
plt.show()

print('Test RMSE: {}'.format(np.sqrt(np.mean((test_pred-Y_test[conv_width-1:])**2))))

train_pred = moving_average(clf.predict(x_train),conv_width)
plt.figure(figsize=(15,15))
plt.plot(train_pred, alpha=0.8, label='Predictions')
plt.plot(Y_train[conv_width-1:],linewidth=7, label='Labels')
plt.title('Training Set Predictions')
plt.xlabel('Samples')
plt.ylabel('Digitized Slope (Degrees)')

plt.show()

print('Train RMSE: {}'.format(np.sqrt(np.mean((train_pred-Y_train[conv_width-1:])**2))))

roll = walk1[:100,8]
peaks, _ = find_peaks(-roll, height=100, width=20)

plt.figure(figsize=(20,10))
plt.plot(roll)
plt.plot(peaks, roll[peaks], 'x')

plt.figure(figsize=(20,20))
sns.heatmap(confusion_matrix(Y_test[conv_width-1:], test_pred, normalize='true'), annot=True)
plt.ylabel('Label')
plt.xlabel('Prediction')
plt.title('Logistic Regression Confusion Matrix')



