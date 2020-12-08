#!/usr/bin/env python

# this script can check the dimension of each case so that 
# we can set a reasonable cropping/padding size

import os
import numpy as np
import zc_function_list as ff
import nibabel as nib
import segcnn

cg = segcnn.Experiment()
fs = segcnn.FileSystem(cg.base_dir, cg.data_dir,cg.local_dir)

patient_list = ff.find_all_target_files(['ucsd_*/*'],cg.local_dir)
x_size = []
y_size = []
z_size = []
for p in patient_list:
    patient_id = os.path.basename(p)
    patient_class = os.path.basename(os.path.dirname(p))
    vol = os.path.join(p,'img-nii-0.625/0.nii.gz')
    vol_data = nib.load(vol).get_fdata()
    dimension = vol_data.shape
    x_size.append(dimension[0])
    y_size.append(dimension[1])
    z_size.append(dimension[-1])
    print(patient_class,patient_id,dimension)
x_size = np.asarray(x_size)
y_size = np.asarray(y_size)
z_size = np.asarray(z_size)
print(np.mean(x_size),np.std(x_size),np.min(x_size),np.max(x_size))
print(np.mean(y_size),np.std(y_size),np.min(y_size),np.max(y_size))
print(np.mean(z_size),np.std(z_size),np.min(z_size),np.max(z_size))
