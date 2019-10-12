#!/usr/bin/env python3
import sys
import numpy as np
import numpy.linalg


class NaiveBayes(object):
    def __init__(self, training_x, training_y, num_class, theta_k, theta_j_k, num_feature):
        self.training_x = training_x
        self.training_y = training_y
        self.num_class = num_class
        self.theta_k = theta_k
        self.theta_j_k = theta_j_k
        self.num_feature = num_feature

    def fit(self):
        number_sample = self.training_y.shape[0]
        for k in range(self.num_class):  # 20 times
            sum_y_in_feature = 0
            for i in range(number_sample):  # 70000 times
                # compute theta_k
                if self.training_y[i][k] == 1:
                    sum_y_in_feature += 1
            self.theta_k[k] = sum_y_in_feature / float(number_sample)
            # print(self.theta_k[k])
        print("start jk")
        for k in range(self.num_class):  # 20 times
            for j in range(self.training_x.shape[1]):  # 74265 times
                sum_theta_jk = 0
                for i in range(number_sample):  # 70000 times
                    # compute theta_jk
                    if self.training_y[i][k] == 1 and self.training_x[i][j] == 1:
                        sum_theta_jk += 1
                self.theta_j_k[j][k] = sum_theta_jk / self.theta_k[k]
                print(self.theta_j_k[j][k])

    def predict(self, validation_x):
        class_prob = []
        for k in range(self.num_class):
            feature_likelihood = 0
            for j in range(self.num_feature):
                feature_likelihood += \
                    validation_x[j] * np.log(self.theta_j_k[j][k]) + (1 - validation_x[j]) * np.log(
                        1 - self.theta_j_k[j][k])
                class_prob = feature_likelihood + np.log(self.theta_k[k])
        return np.argmax(class_prob)
