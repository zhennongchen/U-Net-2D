#!/usr/bin/env python

# System
import os
import glob as gb
import pathlib as plib
import numpy as np
import dvpy as dv
import segcnn
import U_Net_function_list as ff
cg = segcnn.Experiment()

np.random.seed(cg.seed)

#make the directories
os.makedirs(cg.partition_dir, exist_ok = True)
# Create a list of all patients.
patient_list = ff.find_all_target_files(['*/*'],cg.local_dir)
print(patient_list.shape,patient_list[0:10])

# Shuffle the patients.
np.random.shuffle(patient_list)
print(patient_list[0:10])

# Split the list into `cg.num_partitions` (approximately) equal sublists.
partitions = np.array_split(patient_list, cg.num_partitions)
# when set seed = 1, VR data partition result:
# batch 0: Abnormal = 27, Normal = 35; batch 1: 31, 31; batch 2: 24, 38; batch 3: 27,34; batch 4: 31,30

# Save the partitions.
np.save(os.path.join(cg.partition_dir,'partitions_local_adapted.npy'), partitions)

def create_img_lists(imglist):
    partitions = np.load(os.path.join(cg.partition_dir,'partitions_local_adapted.npy'),allow_pickle = True)
    for i, partition in enumerate(partitions):
        if imglist == 'one_time_frame_LV':
            t = [os.path.join(c, 'time_frame_picked_for_pretrained_AI.txt') for c in partition]
            t = [int(open(s, 'r').read()) for s in t]
            segs = [[os.path.join(c, 'seg-nii-1.5-upsample-retouch-adapted-LV', 'pred_s_'+str(f)+'.npy')] for c, f in zip(partition, t)]
            segs = dv.collapse_iterable(segs)
            imgs = [os.path.join(os.path.dirname(os.path.dirname(s)), 'img-nii-0.625-adapted', str(ff.find_timeframe(s,1,'_'))+'.npy') for s in segs]
           
        assert(len(imgs) == len(segs))
        os.makedirs(os.path.join(cg.partition_dir, imglist), exist_ok = True)
        np.save(os.path.join(cg.partition_dir,imglist,'img_list_'+str(i)+'.npy'), imgs)
        np.save(os.path.join(cg.partition_dir,imglist,'seg_list_'+str(i)+'.npy'), segs)
    

# main
create_img_lists('one_time_frame_LV')
