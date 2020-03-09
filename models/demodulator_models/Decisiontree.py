from sklearn.tree import DecisionTreeClassifier
# import visualize
from sklearn.model_selection import train_test_split
import random
import math

import numpy as np
import torch
from collections import deque

from utils import util_modulation
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from utils.util_data import integers_to_symbols, symbols_to_integers

class Decisiontree():
    def __init__(self, *,
                 bits_per_symbol: int,
                 max_amplitude: float = 0.0,
                 max_depth: int,
                 proj_num: int,
                 **kwargs):
        self.name = 'dtree'
        self.model_type = 'demodulator'
        self.bits_per_symbol = bits_per_symbol
        self.clf = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=4, random_state=1)
        self.proj_num = proj_num
        self.X_train = []
        self.y_train = []
        self.a_list = []
        self.b_list = []
        for i in range(proj_num):
            a, b = random.uniform(-1, 1), random.uniform(-1, 1)
            a, b = a / math.sqrt(a ** 2 + b ** 2), b / math.sqrt(a ** 2 + b ** 2)
            self.a_list.append(a)
            self.b_list.append(b)

    def forward(self, symbols: torch.Tensor, **kwargs):

        '''
        Inputs:
        symbols: torch.tensor of type float and shape [n,2] with modulalated symbols
        soft: Bool either True or False
        Output:
        if mode = "prob"
            labels_si_g: torch.tensor of type float and shape [n,2] with probabilities for each demodulated symbol
        else:
            labels_si_g: torch.tensor of type int and shape [n,1] with integer representation of demodulated symbols
        '''
        X_test = symbols.numpy()

        for i in range(self.proj_num):
            pro = np.array([X_test[:, 0] * self.a_list[i] + X_test[:, 1] * self.b_list[i]]).T
            X_test = np.hstack((X_test, pro))

        y_pred = self.clf.predict(X_test)
        y_pred = torch.from_numpy(y_pred)
        return y_pred

    __call__ = forward

    def update(self, signal, true_symbols, times, **kwargs):
        if len(signal.shape) == 2:
            cartesian_points = torch.from_numpy(signal).float()
        elif len(signal.shape) == 1:
            cartesian_points = torch.from_numpy(
                np.stack((signal.real.astype(np.float32), signal.imag.astype(np.float32)), axis=-1))

        X = cartesian_points.numpy()
        y = symbols_to_integers(true_symbols)

        # sliding window
        if times == 0:
            self.X_train = X
            self.y_train = y
        elif times <= 25:
            self.X_train = np.vstack((self.X_train, X))
            self.y_train = np.append(self.y_train, y)
        else:
            self.X_train = np.vstack((self.X_train[signal.shape[0]:, :], X))
            self.y_train = np.append(self.y_train[signal.shape[0]:], y)

        X_ = self.X_train
        for i in range(self.proj_num):
            pro = np.array([self.X_train[:, 0] * self.a_list[i] + self.X_train[:, 1] * self.b_list[i]]).T
            X_ = np.hstack((X_, pro))

        self.clf.fit(X_, self.y_train)
