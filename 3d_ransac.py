import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()
X_train = diabetes.data[:-20, (0,1)]

y_train = diabetes.target[:-20]

ransac = linear_model.RANSACRegressor(
                                        linear_model.LinearRegression()
                                     )

ransac.fit(X_train, y_train)

fig = plt.figure()
plt.clf()

ax = Axes3D(fig)

ax.plot_surface([-5,5],[-5,5], ransac.predict(X_train))