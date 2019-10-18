#!/usr/bin/env python3
import numpy as np


class NaiveBayes(object):
    def __init__(self, training_x, training_y, num_class, theta_k, theta_j_k,
                 num_feature, num_sample):
        self.training_x = training_x
        self.training_y = training_y
        self.num_class = num_class
        self.theta_k = theta_k
        self.theta_j_k = theta_j_k
        self.num_feature = num_feature
        self.num_sample = num_sample

    def fit(self):
        for k in range(self.num_class):  # 20 times
            self.theta_k[k] = ((self.training_y.indices == k).sum() + 1) / float(self.num_sample + 2)
            print(self.theta_k[k])

        for k in range(self.num_class):  # 20 times
            for j in range(self.num_feature):  # 1000 times
                self.theta_j_k[j][k] = \
                    ((self.training_x[self.training_y.indices == k].indices == j).sum() + 1) \
                    / float((self.training_y.indices == k).sum() + 2)
            print(self.theta_j_k[j][k], "new j", j)

    def predict(self, validation_x, feature_name):
        # for every sample in test set, generate prob for 20 classes choose the highest one.
        # return a list of pred_y
        pred_y = ["" for x in range(len(validation_x))]
        i = 0
        for x in validation_x:
            # class_prob = -100000000000
            class_prob = []
            for k in range(self.num_class):
                feature_likelihood = 0
                for j in range(self.num_feature):
                    feature_likelihood += x[j] * np.log(self.theta_j_k[j][k]) + \
                                          (1 - x[j]) * np.log(1 - self.theta_j_k[j][k])
                class_prob.append(feature_likelihood + np.log(self.theta_k[k]))
            pred_y[i] = feature_name[np.argmax(class_prob)]

            print(i, pred_y[i])
            i += 1
        return pred_y

