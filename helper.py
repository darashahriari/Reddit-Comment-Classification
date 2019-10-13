import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Helper:
    def run_pca(self, train, val):
        scaler = StandardScaler()
        train_std = scaler.fit_transform(train)
        val_std = scaler.fit_transform(val) 
        pca = PCA(.95)
        pca.fit(train_std)
        print(pca.n_components_)
        train_img = pca.transform(train_std)
        test_img = pca.transform(val_std)
        return train_img, test_img

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
