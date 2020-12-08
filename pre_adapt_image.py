#!/usr/bin/env python

# this script will adapt the image (crop/pad + normalize + relabel + resize...)
import os
import numpy as np
import dvpy as dv
import segcnn
import segcnn.utils as ut
import zc_function_list as ff
cg = segcnn.Experiment()

# define patient list
patient_list = ff.find_all_target_files(['ucsd_*/*'],cg.base_dir)

for p in patient_list:
    patient_class = os.path.basename(os.path.dirname(p))
    patient_id = os.path.basename(p)
    print(patient_class,patient_id)

    # read es file
    es_file = open(os.path.join(p,'es.txt'),"r")
    es = es_file.read()
    if es[-1] =='\n':
        es = es[0:len(es)-1]
    ed = '0'


    # adapt input image - CT volume
    save_folder = os.path.join(p,'img-nii-0.625-adapted')
    ff.make_folder([save_folder])
    img_list = ff.find_all_target_files([ed+'.nii.gz',es+'.nii.gz'],os.path.join(p,'img-nii-0.625'))
    for i in img_list:
        time = ff.find_timeframe(i,2)
        x = ut.in_adapt(i)
        print(x.shape)
        if cg.normalize == 1:
            print('normalize is done')
            x = ut.normalize_image(x)
        np.save(os.path.join(save_folder,str(time)+'.npy'),x)
    
    # adapt output image - CT segmentation
    save_folder = os.path.join(p,'seg-nii-0.625-adapted')
    ff.make_folder([save_folder])
    seg_list = ff.find_all_target_files([ed+'.nii.gz',es+'.nii.gz'],os.path.join(p,'seg-nii-0.625'))
    for s in seg_list:
        time = ff.find_timeframe(s,2)
        y = ut.out_adapt(s,cg.relabel_LVOT)
        print(y.shape)
        np.save(os.path.join(save_folder,str(time)+'.npy'),y)


        
        







