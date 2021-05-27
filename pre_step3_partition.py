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
patient_list = ff.find_all_target_files(['*/*'],cg.local_dir)  # these are all the patients (with adapted image volumes) you save in the octomore local
#patient_list1 = ff.get_patient_list_from_csv(os.path.join(cg.spreadsheet_dir,'Lead_patient_list.csv'))
patient_list = []
for p in patient_list:
    patient_list.append(os.path.join(cg.local_dir,p[0],p[1]))
patient_list = np.asarray(patient_list)
print(patient_list.shape,patient_list[0:3])

# Randomly Shuffle the patients.
np.random.shuffle(patient_list)
print(patient_list[0:3])

# Split the list into `cg.num_partitions` (approximately) equal sublists.
partitions = np.array_split(patient_list, cg.num_partitions)
# when set seed = 1, VR data partition result:
# batch 0: Abnormal = 27, Normal = 35; batch 1: 31, 31; batch 2: 24, 38; batch 3: 27,34; batch 4: 31,30

# Save the partitions.
np.save(os.path.join(cg.partition_dir,'partitions_local_adapted.npy'), partitions)

# create numpy arrays that save the list of paths for image files and segmentation files in each group
# AI model will read these numpy arrays to know who are in training and who are in validation.
def create_img_lists(imglist):
    partitions = np.load(os.path.join(cg.partition_dir,'partitions_local_adapted.npy'),allow_pickle = True)
    print(partitions.shape)

    for i, partition in enumerate(partitions):
        # define which time frame(s) in each case you will use in the model
        # What I did is to pre-define this/these frames and save in a txt file "time_frame_picked_for_pretrained_AI.txt" in the folder of each case in octomore
        # You can define in your own way
        t = [os.path.join(c, 'time_frame_picked_for_pretrained_AI.txt') for c in partition]
        t = [int(open(s, 'r').read()) for s in t] # t is the frame I define 

        # find the file path of segmentation file of this/these frame(s)
        segs = [[os.path.join(c, 'seg-nii-1.5-upsample-retouch-adapted', 'pred_s_'+str(f)+'.npy')] for c, f in zip(partition, t)] # path of manual segmentation
        segs = dv.collapse_iterable(segs)
        imgs = [os.path.join(os.path.dirname(os.path.dirname(s)), 'img-nii-0.625-adapted', str(ff.find_timeframe(s,1,'_'))+'.npy') for s in segs] # corresponding image file 
           
        assert(len(imgs) == len(segs))
        os.makedirs(os.path.join(cg.partition_dir, imglist), exist_ok = True)
        np.save(os.path.join(cg.partition_dir,imglist,'img_list_'+str(i)+'.npy'), imgs)
        np.save(os.path.join(cg.partition_dir,imglist,'seg_list_'+str(i)+'.npy'), segs)
    

# main
create_img_lists('one_time_frame_4classes')
