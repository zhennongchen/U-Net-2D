#!/usr/bin/env python

## 
# this script can do file copy and removal
##

import os
import numpy as np
import U_Net_function_list as ff
import shutil
import pandas as pd
import segcnn

cg = segcnn.Experiment()



# # file transfer to octomore
# print(cg.local_dir)
# patient_list = ff.get_patient_list_from_csv(os.path.join(cg.spreadsheet_dir,'Final_patient_list_include.csv'))
# for p in patient_list:
#     patient_id = p[1]
#     patient_class = p[0]
#     print(patient_class,patient_id)
    
#     ff.make_folder([os.path.join(cg.local_dir,patient_class),os.path.join(cg.local_dir,patient_class,patient_id)])
    
#     #img_folder = os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-0.625-adapted')
#     #shutil.copytree(img_folder,os.path.join(cg.local_dir,patient_class,patient_id,'img-nii-0.625-adapted'))
    
#     seg_folder = os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-nii-1.5-upsample-retouch-adapted-LV')
#     shutil.copytree(seg_folder,os.path.join(cg.local_dir,patient_class,patient_id,'seg-nii-1.5-upsample-retouch-adapted-LV'))
    
#     #txt_file = os.path.join(cg.seg_data_dir,patient_class,patient_id,'time_frame_picked_for_pretrained_AI.txt')
#     #shutil.copy(txt_file,os.path.join(cg.local_dir,patient_class,patient_id,'time_frame_picked_for_pretrained_AI.txt'))
    