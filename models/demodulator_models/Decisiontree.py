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
from utils.util_data import integers_to_symbols,symbols_to_integers
import copy


a_list = []
b_list = []
for i in range(50):
    a, b = random.uniform(-1, 1), random.uniform(-1, 1)
    a, b = a / math.sqrt(a ** 2 + b ** 2), b / math.sqrt(a ** 2 + b ** 2)
    a_list.append(a)
    b_list.append(b)


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
        self.symbol_map = torch.from_numpy(util_modulation.get_symbol_map(bits_per_symbol=bits_per_symbol)).float()
        self.clf = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=4, random_state=1)
        # self.clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=1)
        # self.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth), n_estimators=200, random_state=1)
        # self.normalize_symbol_map()
        self.proj_num = proj_num
        self.factor = []
        self.times = 0
        self.window_size = 5
        # self.preamble = None
        # self.X = None
        # self.y = None
        self.max_acc_tree = []

    # def normalize_symbol_map(self):
    #     # I and Q separate
    #     if self.max_amplitude > 0.0:
    #         avg_power = torch.mean(torch.sum(self.symbol_map ** 2, dim=-1))
    #         normalization_factor = torch.sqrt(
    #             (torch.relu(avg_power - self.max_amplitude) + self.max_amplitude) / self.max_amplitude)
    #         self.symbol_map = self.symbol_map / normalization_factor

    def forward(self, symbols: torch.Tensor, preamble = None, **kwargs):

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

        #eliminate steps.
        for i in range(self.proj_num):
            # a, b = random.uniform(-1, 1), random.uniform(-1, 1)
            # a, b = a / math.sqrt(a ** 2 + b ** 2), b / math.sqrt(a ** 2 + b ** 2)
            pro = np.array([X_test[:, 0] * a_list[i] + X_test[:, 1] * b_list[i]]).T
            X_test = np.hstack((X_test, pro))


        # print(X_test)
        # y_pred = self.max_acc_tree[0].predict(X_test)
        y_pred =  self.clf.predict(X_test)
        if preamble is not None:
            y_preamble = symbols_to_integers(preamble)
            acc = sum(y_preamble[i]==y_pred[i] for i in range(len(preamble)))/len(preamble)
            # print(acc)
            # if self.max_acc_tree: print('acc_clf:', acc, "acc_max:", self.max_acc_tree[1])
            if not self.max_acc_tree:
                self.max_acc_tree.append(copy.deepcopy(self.clf))
                self.max_acc_tree.append(acc)
            elif self.max_acc_tree[1]<acc:
                self.max_acc_tree[0] = copy.deepcopy(self.clf)
                self.max_acc_tree[1] = acc

        if self.max_acc_tree:
            y_pred = self.max_acc_tree[0].predict(X_test)

        # np.savetxt('X_test.txt', X_test)
        # np.savetxt('Y_pred.txt', y_pred)

        # X_test_com = X_test[:, 0] + 1j * X_test[:, 1]
        y_pred = torch.from_numpy(y_pred)
        # print(y_pred)
        return y_pred

    __call__ = forward

    def update(self, signal, true_symbols, **kwargs):
        # self.times = self.times + 1
        # w = 1
        # if self.times % 50 == 0:
        #     w = self.times/50

        # print('X_train', signal)
        # print('Y_train', signal)
        # np.savetxt('X_train.txt', signal)
        # np.savetxt('Y_train.txt', true_symbols)
        # filename = 'train_data.txt'
        # with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        #     f.write(signal)
        #     f.write(true_symbols)

        if len(signal.shape) == 2:
            cartesian_points = torch.from_numpy(signal).float()
        elif len(signal.shape) == 1:
            cartesian_points = torch.from_numpy(
                np.stack((signal.real.astype(np.float32), signal.imag.astype(np.float32)), axis=-1))
        X = cartesian_points.numpy()
        y = symbols_to_integers(true_symbols)

        # if self.times == 1:
        #     self.X = cartesian_points.numpy()
        #     self.y = symbols_to_integers(true_symbols)
        # elif self.times <= self.window_size:
        #     print(self.X.shape)
        #     print(cartesian_points.numpy().shape)
        #     self.X = np.vstack((self.X, cartesian_points.numpy()))
        #     self.y = np.vstack((self.y, symbols_to_integers(true_symbols)))
        # else:
        #     self.X = np.vstack((self.X[signal.shape[0]:, :], cartesian_points.numpy()))
        #     self.y = np.vstack((self.y[signal.shape[0]:, :], symbols_to_integers(true_symbols)))

        # y = true_symbols

        # pro = np.zeros((n_sample, 1))
        for i in range(self.proj_num):
            pro = np.array([X[:, 0] * a_list[i] + X[:, 1] * b_list[i]]).T
            X = np.hstack((X, pro))

        # self.clf.n_estimators += 1
        self.clf.fit(X, y)
        # y_pred = self.clf.predict(X)
        # acc = sum(y[i]==y_pred[i] for i in range(len(y)))/len(y)
        # print(acc)
        # if not self.max_acc_tree:
        #     self.max_acc_tree.append(copy.deepcopy(self.clf))
        #     self.max_acc_tree.append(acc)
        # elif acc>self.max_acc_tree[1]:
        #     self.max_acc_tree[0] = copy.deepcopy(self.clf)
        #     self.max_acc_tree[1] = acc

