#!/usr/bin/env python


import os
import numpy as np
import U_Net_function_list as ff
import nibabel as nb
import pandas as pd
import segcnn

cg = segcnn.Experiment()

# find patient list
patient_list = ff.find_all_target_files(['*/*'],cg.seg_data_dir)
print(patient_list.shape)

result = []
for p in patient_list:
    patient_id = os.path.basename(p)
    patient_class = os.path.basename(os.path.dirname(p))
    print(patient_class,patient_id)

    # read time frame file
    f = open(os.path.join(p,'time_frame_picked_for_pretrained_AI.txt'),"r")
    t = int(f.read())

    # read threshold setting:
    threshold_file = os.path.join(p,'threshold.txt')
    if os.path.isfile(threshold_file) == 0:
        raise ValueError('no threshold txt file')
    else:
        threshold = open(threshold_file,"r")
        threshold = int(threshold.read())
        print(threshold)

    # load image and seg
    imgfile = os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-0.625',str(t)+'.nii.gz')
    segfile = os.path.join(p,'seg-pred-1.5-upsample-retouch','pred_s_'+str(t)+'.nii.gz')
    if os.path.isfile(imgfile) == 1 and os.path.isfile(segfile) == 1:
        img = nb.load(imgfile).get_fdata()
        seg = nb.load(segfile).get_fdata()
        _,LV_pixels = ff.count_pixel(seg,1)
        LV_pixels = np.asarray(LV_pixels)

        # find the center of geometry
        x = LV_pixels[:,0]; x_mean = int(np.mean(x))
        y = LV_pixels[:,1]; y_mean = int(np.mean(y))
        z = LV_pixels[:,2]; z_mean = int(np.mean(z))

        # set a small ROI
        x_range = 40; z_range = 20
        seg_roi = seg[x_mean-x_range:x_mean+x_range, y_mean-x_range:y_mean+x_range, z_mean-z_range:z_mean+z_range]
        img_roi = img[x_mean-x_range:x_mean+x_range, y_mean-x_range:y_mean+x_range, z_mean-z_range:z_mean+z_range]    
        
        # calculate intensity stat
        _,LV_pixels = ff.count_pixel(seg_roi,1)
        intensity_list = []
        for L in LV_pixels:
            intensity_list.append(img_roi[L[0],L[1],L[2]])
        mean_value = np.mean(np.asarray(intensity_list))
        std_value = np.std(np.asarray(intensity_list))
    else:
        print('no image or segmentation')
        mean_value =''
        std_value = ''

    result.append([patient_class,patient_id,threshold,mean_value,std_value])


df = pd.DataFrame(result,columns = ['Patient_Class','Patient_ID','threshold_set','mean_I','std_I'])
df.to_excel(os.path.join(cg.spreadsheet_dir,'LV_intensity_list.xlsx'),index = True)



    





