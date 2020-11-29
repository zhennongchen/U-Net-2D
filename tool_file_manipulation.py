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

## define patient_list
#patient_list = ff.find_all_target_files(['*/*'],cg.raw_dir)



# # make folder
# for p in patient_list:
#     patient_id = os.path.basename(p)
#     patient_class = os.path.basename(os.path.dirname(p))

#     print(patient_class,patient_id)
#     ff.make_folder([os.path.join(cg.base_dir,patient_class),os.path.join(cg.base_dir,patient_class,patient_id)])


# file copy
folder="/Data/McVeighLabSuper/projects/Zhennong/AI/CNN/all-classes-all-phases-1.5/"
es_file_list = ff.find_all_target_files(['*/*/es.txt'],folder)
print(es_file_list)
print(es_file_list.shape)
for e in es_file_list:
    patient_id = os.path.basename(os.path.dirname(e))
    patient_class = os.path.basename(os.path.dirname(os.path.dirname(e)))

    print(patient_class,patient_id)
    shutil.copy(e,os.path.join(cg.base_dir,patient_class,patient_id,'es.txt'))