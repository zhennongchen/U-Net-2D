#!/usr/bin/env python

import dvpy as dv
import dvpy.tf
import segcnn
import nibabel as nib
import os
import numpy as np
import math
import pandas as pd
import U_Net_function_list as ff
cg = segcnn.Experiment()
def read_nib(file):
    img = nib.load(file)
    data = img.get_fdata()
    shape = data.shape
    return img,data,shape

pred_seg_file = 'pred_s_'
excel_filename = 'UNet_VR_1tf_2class_segmentation_volumetric_evaluation.xlsx'
structure_list = [('LV',1),('LA',2),('LAA',3),('LVOT',4),('Aorta',5),('PV',6)] #(chamber,value in the seg)
structure_choice = [0]

column_list = ['Patient_class','Patient_ID','batch']

patient_list = ff.get_patient_list_from_csv(os.path.join(cg.spreadsheet_dir,'Final_patient_list_include.csv'))
partition_file_path = os.path.join(cg.partition_dir,'partitions_local_adapted.npy')


result = []
count = 0
for p in patient_list:
  patient_id = p[1]
  patient_class = p[0]
  patient_batch = ff.locate_batch_num_for_patient(patient_class,patient_id,partition_file_path)
  print(patient_class,patient_id,patient_batch)

  # read time frame picked
  f = open(os.path.join(cg.seg_data_dir,patient_class,patient_id,'time_frame_picked_for_pretrained_AI.txt'),"r")
  t = int(f.read())

  # make sure the prediction segmentation exists
  if os.path.exists(os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-pred-0.625-LV',pred_seg_file + str(t)+'.nii.gz')) != 1 or os.path.exists(os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-pred-1.5-upsample-retouch',pred_seg_file+str(t)+'.nii.gz')) != 1:
    #print('no prediction or ground truth segmentation in this patient')
    continue
  
  patient_result_list = [patient_class,patient_id,patient_batch]


  # ground truth segmentation
  seg_t,data_t,_ = read_nib(os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-pred-1.5-upsample-retouch',pred_seg_file+str(t)+'.nii.gz'))
    
  # predicted segmentation
  seg_p,data_p,_ = read_nib(os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-pred-0.625-LV',pred_seg_file+str(t)+'.nii.gz'))


  for choice in structure_choice:
    structure = structure_list[choice][0]
    target_val = structure_list[choice][1]
    if count == 0:
      column_list.append(structure+'_DICE')
      #column_list.extend((structure+'_DICE_ED',structure+'_DICE_ES',structure+'_EDV_truth',structure+'_ESV_truth',structure+'_EF_truth',
      #structure+'_EDV_pred',structure+'_ESV_pred',structure+'_EF_pred'))

    # volumetric calculation
    DICE = ff.DICE(data_t,data_p,target_val)
    # DICE_ES = ff.DICE(data_ES_t,data_ES_p,target_val)
    # EDV_truth,_ = ff.count_pixel(data_ED_t,target_val)
    # ESV_truth,_ = ff.count_pixel(data_ES_t,target_val)
    # if EDV_truth == 0:
    #   EF_truth = 0
    # else:
    #   if structure == 'LV':
    #     EF_truth = (EDV_truth - ESV_truth) / EDV_truth
    #   else:
    #     EF_truth = (ESV_truth - EDV_truth) / ESV_truth
    # EDV_pred,_ = ff.count_pixel(data_ED_p,target_val)
    # ESV_pred,_ = ff.count_pixel(data_ES_p,target_val)
    # if EDV_pred == 0:
    #   EF_pred = 0
    # else:
    #   if structure == 'LV':
    #     EF_pred = (EDV_pred - ESV_pred) / EDV_pred
    #   else:
    #     EF_pred = (ESV_pred - EDV_pred) / ESV_pred
   
    #patient_measurements = [DICE_ED,DICE_ES,EDV_truth,ESV_truth,EF_truth,EDV_pred,ESV_pred,EF_pred]
    patient_measurements = [DICE]
    print(DICE)
    patient_result_list = patient_result_list + patient_measurements
  
  count = count + 1
  result.append(patient_result_list)

  
# save into dataframe
df = pd.DataFrame(result,columns = column_list)
df.to_excel(os.path.join(cg.spreadsheet_dir,excel_filename),index = False)