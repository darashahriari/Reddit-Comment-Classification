import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA

class Helper:
    def run_lsa(self, X_train):
        scaler = StandardScaler(with_mean=False)
        train_std = scaler.fit_transform(X_train)
        lsa = TruncatedSVD(n_iter=100, n_components=100)
        lsa.fit(train_std)
        return lsa

    def plot(self, x, y, x_label, y_label, title):
        plt.figure()
        plt.plot(x,y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.suptitle(title)
    
    def show(self):
        plt.show()

    def correlation_heatmap(self, train, y):
        total = np.append(train,y,axis=1)
        total = pd.DataFrame(total)
        total.columns = []
        correlations = total.corr()

        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(correlations, annot=True, cmap="YlGnBu")
        plt.show()  

    def generate_prediction_csv(self, pred):
        df = pd.DataFrame({'Category': pred})
        df.to_csv(index=True, path_or_buf='validation.csv', index_label='Id')
