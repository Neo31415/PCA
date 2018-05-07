from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def PCA():
	img = mpimg.imread('imagep.png')
	print("Real Image Shape:", img.shape)
	a = img.shape[0]
	b = img.shape[1]
	c = img.shape[2]

	img_r = np.reshape(img, (a, b*c))
	print("Reshaped Image Shape:", img_r.shape)
	ipca = RandomizedPCA(1000).fit(img_r)
	img_c = ipca.transform(img_r)
	print(img_c.shape)
	print(np.sum(ipca.explained_variance_ratio_))
	temp = ipca.inverse_transform(img_c)
	temp = np.reshape(temp, (a, b, c))
	print(temp.shape, a, b, c)

	plt.axis('off')
	plt.imshow(temp)
	#plt.imshow(temp)
	plt.show()