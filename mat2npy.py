import numpy as np
import os
import scipy.io as sio

from scipy.signal import resample


data_dir = './data/mat/cleaned'
save_dir = './data/npy'
fs = 100.0

data = []
for f in os.listdir(data_dir):
    print(f)
    mat = sio.loadmat(os.path.join(data_dir, f))
    data.append(mat[list(mat.keys())[-1]][0])

data_re = []
for sess in data:

    accel = sess[0]
    angvel = sess[1]
    ori = sess[2]
    slope = sess[3]
    print(ori.shape)
    slope_re = resample(slope, ori.shape[0])
    x = np.hstack((accel, angvel, ori))
    y = -slope_re
    data_re.append(np.column_stack((x[10000:-2000,:],y[10000:-2000,1])))


#bins = [0, -1, -2, -3, -4, -5, -6, -7, -8]
#walk1 = np.column_stack([data_re[0][:,:9], np.digitize(data_re[0][:,9], bins)])
#walk2 = np.column_stack([data_re[1][:-22500,:9], np.digitize(data_re[1][:-22500,9], bins)])
#walk_regr = np.column_stack([data_re[1][-22500:,:9], data_re[1][-22500:,9]])

walk1 = data_re[0][:,:]
walk2 = data_re[1][:-22500,:]

np.save(os.path.join(save_dir, 'walk1_regr'),walk1)
np.save(os.path.join(save_dir, 'walk2_regr'), walk2)
