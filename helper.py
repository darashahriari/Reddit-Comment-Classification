import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Helper:
	def run_pca(self, train, val):
		scaler = StandardScaler()
		train_std = scaler.fit_transform(train)
		val_std = scaler.fit_transform(val)	
		pca = PCA().fit(train_std)

	def plot(self, x, y, x_label, y_label, title):
		plt.figure()
		plt.plot(x,y)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.suptitle(title)
	
	def show(self):
		plt.show()	