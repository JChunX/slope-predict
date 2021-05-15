import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import resample

data_dir = './data/npy'

data_list = []
data_list.append(np.load(os.path.join(data_dir, 'walk1_regr.npy')))
data_list.append(np.load(os.path.join(data_dir, 'walk2_regr.npy')))

resample_range = range(-40000, 40000, 2000)
for i, data in enumerate(data_list):

    for flip in [False]:
        for r in resample_range:
            d = np.copy(data)
            d = resample(d, int(len(d)/3))
            d = resample(d, len(d) - r)
            if flip:
                d[:,:2] = -d[:,:2]
                d[:,3:5] = -d[:,3:5]
                d[:,7] = -d[:,7]
                np.save(os.path.join(
                    data_dir,'augment','walk{0}_resample_{1}.npy'.format(i,r)), d)
            else:
                np.save(os.path.join(
                    data_dir,'augment','walk{0}_resample_{1}_flip.npy'.format(i,r)), d)


