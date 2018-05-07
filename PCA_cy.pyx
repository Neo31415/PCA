from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def PCA_FUN(img):
	fig = plt.figure()
	comp = 64
	print("Real Image Shape:", img.shape)
	a = img.shape[0]
	b = img.shape[1]
	c = img.shape[2]

	img_r = np.reshape(img, (a, b*c))
	print("Reshaped Image Shape:", img_r.shape)
	ipca = RandomizedPCA(comp).fit(img_r)
	img_c = ipca.transform(img_r)
	print(img_c.shape)
	print(np.sum(ipca.explained_variance_ratio_))
	temp = ipca.inverse_transform(img_c)
	temp = np.reshape(temp, (a, b, c))
	print(temp.shape, a, b, c)

	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

	ax1.axis('off')
	ax2.axis('off')
	ax1.set_title('Component='+str(comp))
	ax2.set_title('Original Image')
	ax1.imshow(temp)
	ax2.imshow(img)
	#plt.imshow(temp)
	plt.show()