#!/usr/bin/env python

import dvpy as dv
import dvpy.tf
import segcnn
import nibabel as nib
import os
import numpy as np
import math
import pandas as pd
import zc_function_list as ff
cg = segcnn.Experiment()
def read_nib(file):
    img = nib.load(file)
    data = img.get_fdata()
    shape = data.shape
    return img,data,shape

pred_seg_file = 'UNet_2D_s_'
excel_filename = 'UNet_2D_segmentation_volumetric_evaluation.xlsx'
structure_list = [('LV',1),('LA',2),('LAA',3),('LVOT',4),('Aorta',5),('PV',6)] #(chamber,value in the seg)
structure_choice = [0]

column_list = ['Patient_class','Patient_ID','batch']
    # 'LV_DICE_ED','LV_DICE_ES',
    # 'LV_EDV_truth','LV_ESV_truth','LV_EF_truth',
    # 'LV_EDV_pred','LV_ESV_pred','LV_EF_pred',
    # 'LA_DICE_ED','LA_DICE_ES',
    # 'LA_EDV_truth','LA_ESV_truth','LA_EF_truth',
    # 'LA_EDV_pred','LA_ESV_pred','LA_EF_pred',
    # 'LVOT_DICE_ED','LVOT_DICE_ES',
    # 'LVOT_EDV_truth','LVOT_ESV_truth','LVOT_EF_truth',
    # 'LVOT_EDV_pred','LVOT_ESV_pred','LVOT_EF_pred']


patient_list = ff.find_all_target_files(['ucsd_*/*'],cg.base_dir)
partition_file_path = os.path.join(cg.data_dir,'partitions_U2.npy')


result = []
count = 0
for p in patient_list:
  patient_id = os.path.basename(p)
  patient_class = os.path.basename(os.path.dirname(p))
  patient_batch = ff.locate_batch_num_for_patient(patient_class,patient_id,partition_file_path)
  print(patient_class,patient_id,patient_batch)

  # make sure the prediction segmentation exists
  if os.path.exists(os.path.join(p,'seg-pred',pred_seg_file + '0.nii.gz')) != 1 or os.path.exists(os.path.join(p,'seg-nii-sm','0.nii.gz')) != 1:
    print('no prediction or ground truth segmentation in this patient')
    continue
  
  patient_result_list = [patient_class,patient_id,patient_batch]

  # find ES and ED time frame
  es_file = open(os.path.join(p,'es.txt'),"r")
  es = es_file.read()
  if es[-1] =='\n':
    es = es[0:len(es)-1]
  ed = '0'

  # ground truth segmentation
  seg_ED_t,data_ED_t,_ = read_nib(os.path.join(p,'seg-nii-sm',ed+'.nii.gz'))
  seg_ES_t,data_ES_t,_ = read_nib(os.path.join(p,'seg-nii-sm',es+'.nii.gz'))
    
  # predicted segmentation
  seg_ED_p,data_ED_p,_ = read_nib(os.path.join(p,'seg-pred',pred_seg_file+ed+'.nii.gz'))
  seg_ES_p,data_ES_p,_ = read_nib(os.path.join(p,'seg-pred',pred_seg_file+es+'.nii.gz'))

  for choice in structure_choice:
    structure = structure_list[choice][0]
    target_val = structure_list[choice][1]
    if count == 0:
      column_list.extend((structure+'_DICE_ED',structure+'_DICE_ES',structure+'_EDV_truth',structure+'_ESV_truth',structure+'_EF_truth',
      structure+'_EDV_pred',structure+'_ESV_pred',structure+'_EF_pred'))

    # volumetric calculation
    DICE_ED = ff.DICE(data_ED_t,data_ED_p,target_val)
    DICE_ES = ff.DICE(data_ES_t,data_ES_p,target_val)
    EDV_truth,_ = ff.count_pixel(data_ED_t,target_val)
    ESV_truth,_ = ff.count_pixel(data_ES_t,target_val)
    if EDV_truth == 0:
      EF_truth = 0
    else:
      if structure == 'LV':
        EF_truth = (EDV_truth - ESV_truth) / EDV_truth
      else:
        EF_truth = (ESV_truth - EDV_truth) / ESV_truth
    EDV_pred,_ = ff.count_pixel(data_ED_p,target_val)
    ESV_pred,_ = ff.count_pixel(data_ES_p,target_val)
    if EDV_pred == 0:
      EF_pred = 0
    else:
      if structure == 'LV':
        EF_pred = (EDV_pred - ESV_pred) / EDV_pred
      else:
        EF_pred = (ESV_pred - EDV_pred) / ESV_pred
   
    patient_measurements = [DICE_ED,DICE_ES,EDV_truth,ESV_truth,EF_truth,EDV_pred,ESV_pred,EF_pred]
    
    patient_result_list = patient_result_list + patient_measurements
  
  count = count + 1
  result.append(patient_result_list)


# save into dataframe
df = pd.DataFrame(result,columns = column_list)
df.to_excel(os.path.join(cg.data_dir,'prediction_Excel',excel_filename),index = False)