import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import resample

data_dir = './data/npy'

data = []
data.append(np.load(os.path.join(data_dir, 'walk1.npy')))
data.append(np.load(os.path.join(data_dir, 'walk2.npy')))

resample_range = range(-80000, 80000, 10000)
for i, d in enumerate(data):
    for flip in [True, False]:
        if flip:
            d[:,:2] = -d[:,:2]
            d[:,3:5] = -d[:,3:5]
            d[:,7] = -d[:,7]
        for r in resample_range:
            d = resample(d, len(d) - r)
            if flip:
                np.save(os.path.join(
                    data_dir,'augment','walk{0}_resample_{1}.npy'.format(i,r)), d)
            else:
                np.save(os.path.join(
                    data_dir,'augment','walk{0}_resample_{1}_flip.npy'.format(i,r)), d)


