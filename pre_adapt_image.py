#!/usr/bin/env python

# this script will adapt the image (crop/pad + normalize + relabel + resize...)
import os
import numpy as np
import dvpy as dv
import segcnn
import segcnn.utils as ut
import U_Net_function_list as ff
cg = segcnn.Experiment()

# define patient list
patient_list = ff.get_patient_list_from_csv(os.path.join(cg.spreadsheet_dir,'Final_patient_list_include.csv'))
print(len(patient_list))

for p in patient_list:
    patient_class = p[0]
    patient_id = p[1]

    # read time frame file
    t_file = open(os.path.join(cg.seg_data_dir,patient_class,patient_id,'time_frame_picked_for_pretrained_AI.txt'),"r")
    t = t_file.read()
    if t[-1] =='\n':
        t = t[0:len(t)-1]

    print(patient_class,patient_id)
  

    # adapt input image - CT volume
    save_folder = os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-0.625-adapted')
    ff.make_folder([save_folder])
    img_list = ff.find_all_target_files(['*.nii.gz'],os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-0.625'))
    for i in img_list:
        time = ff.find_timeframe(i,2)
        x = ut.in_adapt(i)
        if cg.normalize == 1:
            print('normalize is done')
            x = ut.normalize_image(x)
        np.save(os.path.join(save_folder,str(time)+'.npy'),x)
    break
    
    # adapt output image - CT segmentation
    # save_folder = os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-nii-1.5-upsample-retouch-adapted-LV')
    # ff.make_folder([save_folder])
    # seg_list = ff.find_all_target_files(['pred_s_'+ t +'.nii.gz'],os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-pred-1.5-upsample-retouch'))
    # for s in seg_list:
    #     #time = ff.find_timeframe(s,2)
    #     y = ut.out_adapt(s,cg.relabel_LVOT)
    #     print(y.shape)
    #     np.save(os.path.join(save_folder,'pred_s_'+str(t)+'.npy'),y)




        
        







