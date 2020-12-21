#!/usr/bin/env python

# this script can check the dimension of each case so that 
# we can set a reasonable cropping/padding size

import os
import numpy as np
import U_Net_function_list as ff
import nibabel as nib
import segcnn

cg = segcnn.Experiment()

patient_list = ff.get_patient_list_from_csv(os.path.join(cg.spreadsheet_dir,'Final_patient_list_include.csv'))
print(len(patient_list))

x_size = []
y_size = []
z_size = []
for p in patient_list:
    patient_id = p[1]
    patient_class = p[0]
    vol = os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-0.625/0.nii.gz')
    vol_data = nib.load(vol).get_fdata()
    dimension = vol_data.shape
    x_size.append(dimension[0])
    y_size.append(dimension[1])
    z_size.append(dimension[-1])
    print(patient_class,patient_id,dimension)
x_size = np.asarray(x_size)
y_size = np.asarray(y_size)
z_size = np.asarray(z_size)
print(np.mean(x_size),np.std(x_size),np.median(x_size),np.min(x_size),np.max(x_size))
print(np.mean(y_size),np.std(y_size),np.median(y_size),np.min(y_size),np.max(y_size))
print(np.mean(z_size),np.std(z_size),np.median(z_size),np.min(z_size),np.max(z_size))



# for VR dataset
# x_dim: mean - 358, median - 352, min - 240, max - 656
# y_dim: mean - 358, median - 352, min - 240, max - 656
# z_dim: mean - 262, median - 356, min - 192, max - 488