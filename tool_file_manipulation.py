#!/usr/bin/env python

## 
# this script can do file copy and removal
##

import os
import numpy as np
import zc_function_list as ff
import shutil
import pandas as pd
import segcnn

cg = segcnn.Experiment()
fs = segcnn.FileSystem(cg.base_dir, cg.data_dir,cg.local_dir)


# # file transfer to octomore
print(cg.local_dir)
patient_list = ff.find_all_target_files(['ucsd_*/*'],cg.base_dir)
for p in patient_list:
    patient_id = os.path.basename(p)
    patient_class = os.path.basename(os.path.dirname(p))
    img_folder = os.path.join(p,'img-nii-0.625-adapted')
    shutil.copytree(img_folder,os.path.join(cg.local_dir,patient_class,patient_id,'img-nii-0.625-adapted'))
    seg_folder = os.path.join(p,'seg-nii-0.625-adapted')
    shutil.copytree(seg_folder,os.path.join(cg.local_dir,patient_class,patient_id,'seg-nii-0.625-adapted'))

#folder = os.path.join(cg.data_dir,'ED_ES')
#shutil.copytree(folder,os.path.join(cg.local_dir,'ED_ES'))  