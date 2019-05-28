from mpl_toolkits.mplot3d import Axes3D

import os
import plyfile
import matplotlib.pyplot as plt
from ransac import ransac, LinearLeastSquaresModel

def read_ply(path):
    plydata = plyfile.PlyData.read(path)
    xx = []
    yy = []
    zz = []
    for i in plydata.elements[0].data[:100000]:
        xx.append(i[0])
        yy.append(i[1])
        zz.append(i[2])

    return (zz, yy, zz)

def run_ransac(data):
    data
    pass

if __name__ == '__main__':
    path = 'C:\\Users\\prajw\\Documents\\Mbzirc\\PcCluster\\data\\test\\1.ply'
    plydata = plyfile.PlyData.read(path)
    # x, y, z = read_ply(path)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(plydata.elements[0].data['x'], plydata.elements[0].data['y'], plydata.elements[0].data['z'], s=0.2, label='points in (x,z)')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


