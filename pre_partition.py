#!/usr/bin/env python

# System
import os
import glob as gb
import pathlib as plib
import numpy as np
import dvpy as dv
import segcnn
import zc_function_list as ff
cg = segcnn.Experiment()
fs = segcnn.FileSystem(cg.base_dir, cg.data_dir,cg.local_dir)

np.random.seed(cg.seed)

#make the directories
os.makedirs(cg.data_dir, exist_ok = True)
# Create a list of all patients.
patient_list = ff.find_all_target_files(['ucsd_*/*'],cg.local_dir)
# Shuffle the patients.
np.random.shuffle(patient_list)
print(patient_list)

# Split the list into `cg.num_partitions` (approximately) equal sublists.
partitions = np.array_split(patient_list, cg.num_partitions)

# Save the partitions.
np.save(fs.partitions('partitions_2D_adapted.npy'), partitions)

def create_img_lists(imglist,suffix):
    partitions = np.load(fs.partitions('partitions_2D_adapted.npy'),allow_pickle = True)
    for i, partition in enumerate(partitions):
        if imglist == 'ALL_SEGS':
            continue
            #segs = [gb.glob(os.path.join(c, cg.seg_dir, fs.img('*'))) for c in partition]
            #segs = dv.collapse_iterable(segs)
            #imgs = [os.path.join(os.path.dirname(os.path.dirname(s)), cg.img_dir, os.path.basename(s)) for s in segs]
      
        elif imglist == 'ED_ES':
            es = [os.path.join(c, 'es.txt') for c in partition]
            es = [int(open(s, 'r').read()) for s in es]
            segs = [[os.path.join(c, cg.seg_dir, fs.img(0)), os.path.join(c, cg.seg_dir, fs.img(f))] for c, f in zip(partition, es)]
            segs = dv.collapse_iterable(segs)
            imgs = [os.path.join(os.path.dirname(os.path.dirname(s)), cg.img_dir, os.path.basename(s)) for s in segs]

        elif imglist == 'ED_ES_adapted':
            es = [os.path.join(c, 'es.txt') for c in partition]
            es = [int(open(s, 'r').read()) for s in es]
            segs = [[os.path.join(c, cg.seg_dir, '0.npy'), os.path.join(c, cg.seg_dir, str(f)+'.npy')] for c, f in zip(partition, es)]
            segs = dv.collapse_iterable(segs)
            imgs = [os.path.join(os.path.dirname(os.path.dirname(s)), cg.img_dir, os.path.basename(s)) for s in segs]
           
        assert(len(imgs) == len(segs))
        os.makedirs(os.path.join(cg.data_dir, imglist+suffix), exist_ok = True)
        np.save(fs.img_list(i, imglist+suffix), imgs)
        np.save(fs.seg_list(i, imglist+suffix), segs)
    

# main
create_img_lists('ED_ES_adapted','')
