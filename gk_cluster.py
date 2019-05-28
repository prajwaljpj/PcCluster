import numpy as np
#import matplotlib.pyplot as plt
import skfuzzy as fuzz
import cv2
import scipy.io as sio

def random_color():
	#rgbl=[255,0,0]
	#random.shuffle(rgbl)
	return list(np.random.choice(range(256), size=3))



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

img = cv2.imread('cube.jpg',0)
img = image_resize(img,width=2400//4)

#img = cv2.imread('temp_crop.png',0)

x,y = img.shape
z = 3
#img_2d = img.reshape(x*y)

feature_map = np.zeros((x,y,3))
for i in range(x):
	for j in range(y):
		feature_map[i][j] = np.array([i,j,img[i][j]])

img_2d = feature_map.reshape(x*y,3)


#img_2d = sio.loadmat('img_2d.mat')
#img_2d = img_2d['arr']
#x,y,z = 720,1280,3

#Initial params
X = img_2d.copy()
X = X.astype(np.float32)

# Number of Clusters
c = 10

# Maximum number of iterations
MAX_ITER = 15

# Number of data points
N,n = X.shape

# Fuzzy parameter
m = 2.00
e = 1e-4

rho = np.ones((1,c))
gamma = 0
beta = 1e15
X1 = np.ones((N,1));

def get_fuzzy_partition_matrix(X,c,m):
	mm = np.mean(X,0) #Columnwise mean
	mm = np.expand_dims(mm,1)

	#Max columnwise
	aa = np.max(np.abs(X - np.ones((N,1)).dot(mm.T)),0)
	aa = np.expand_dims(aa,1)

	#v = 2*(np.ones((c,1))*aa)* (np.random.random((c,n)) - 0.5) + np.ones((c,1))*mm

	# .* is elemwise and * is np.dot in Matlab
	aa =aa.T

	v = 2*np.ones((c,1)).dot(aa)
	v = v*np.random.random((c,n)) - 0.5
	v = v + np.ones((c,1)).dot(mm.T)

	d = []

	for j in range(c):
		xv = X - X1.dot(np.expand_dims(v[j],0)) 
		d.append(np.sum(np.power(xv,2),axis=1))

	d = np.array(d)
	d = d.T

	d = (d + 1e-10) ** (-1/(m-1))
	f0 = d / (np.expand_dims(np.sum(d,1),1).dot(np.ones((1,c))))

	return f0,d

#f0,d = get_fuzzy_partition_matrix(X,c,m)

cntr, u_orig, u0, d, jm, p, fpc = fuzz.cluster.cmeans(img_2d.T,c,m,error=0.005, maxiter=20,metric='mahalanobis') 

f0 = u_orig.T 
d = d.T

f = np.zeros((N,c))

iteration = 0

#NUMPY COV EXPECTS TRANSPOSE.
A0 = np.eye(n)*np.linalg.det(np.cov(X.T))**(1/n)

J = [] #OBJECTIVE FUNCTION
while np.max(f0-f) > e:
	iteration += 1
	f = f0

	#Calculate centers

	fm = f**m
	sumf = np.sum(fm,0)
	sumf = np.expand_dims(sumf,0)

	#v = fm.T*X / (sumf.T*np.ones((1,n)))
	v = fm.T.dot(X) / sumf.T.dot(np.ones((1,n)))

	for j in range(c):
		xv = X - X1.dot(np.expand_dims(v[j],0)) 
		
		#Covariance matrix for each cluster
		A = np.ones((n,1)).dot(np.expand_dims(fm[:,0],1	).T) * xv.T
		A = A.dot(xv)/sumf[:,j]
		A = (1-gamma) * A + gamma*(A0**(1/n))
		
		
		if np.linalg.cond(A) > beta:
			ev,ed = np.linalg.eig(A)
			edmax = max(np.diag(ed))
			ev = np.expand_dims(ev,1)
			z = np.diag(np.expand_dims(np.diag(ed),1))
			z = np.expand_dims(z,1)
			A = ev.dot(z).dot(np.linalg.inv(ev))
		
		#Distance calculations
		M = (1/np.linalg.det(np.linalg.pinv(A)) / rho[:,j])**(1/n)*np.linalg.pinv(A)
		d[:,j] = np.sum((xv.dot(M)*xv),1)


	distout = np.sqrt(d)
	J.append(np.sum(np.sum(f0*d,0),0))

	#Update fuzzy matrix f0
	d = (d + 1e-10) ** (-1/(m-1))
	f0 = d / (np.expand_dims(np.sum(d,1),1).dot(np.ones((1,c))))
	

labels = np.apply_along_axis(np.argmax,1,f0)
labels = np.expand_dims(labels,1)

cluster_img = np.repeat(labels[:, :, np.newaxis], 3, axis=2)
cluster_img = np.squeeze(cluster_img)

colors = {i:random_color() for i in range(c+1)}


cluster_img = cluster_img.reshape(x,y,3)

for i in range(cluster_img.shape[0]):
	for j in range(cluster_img.shape[1]):
		col = colors[cluster_img[i][j][0]]
		cluster_img[i][j] = col

	#for j in range(cluster_img.shape[1]):
	#	cluster_img[i][j] = colors[cluster_img[i][j][0]]

cluster_img = cluster_img.astype(np.uint)
#sio.imsave('temp.png',cluster_img)
cv2.imwrite('temp_gk.png',cluster_img)



