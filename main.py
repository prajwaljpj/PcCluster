import os
import plyfile
import plotly
import plotly.graph_objs as go


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

if __name__ == '__main__':
    path = 'C:\\Users\\prajw\\Documents\\Mbzirc\\PcCluster\\data\\test\\1.ply'
    plydata = plyfile.PlyData.read(path)
    # x, y, z = read_ply(path)

    plotly.offline.plot({
    "data": [go.Scatter3d(x=plydata.elements[0].data['x'], y=plydata.elements[0].data['y'], z=plydata.elements[0].data['z'], mode='markers', marker=dict(size=1))],
    "layout": go.Layout(title="scatter input")
	}, auto_open=True)


