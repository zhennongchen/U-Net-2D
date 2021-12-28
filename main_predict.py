#!/usr/bin/env python
# this script can predict for all timeframes
# run by python main_predict.py

# System
import argparse
import os

# Third Party
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.models import Model
from keras.layers import Input, \
                         Conv1D, Conv2D, Conv3D, \
                         MaxPooling1D, MaxPooling2D, MaxPooling3D, \
                         UpSampling1D, UpSampling2D, UpSampling3D, \
                         Reshape, Flatten, Dense
from keras.initializers import Orthogonal
from keras.regularizers import l2
import nibabel as nb
from sklearn.metrics import mean_squared_error  
import math
# Internal
from segcnn.generator import ImageDataGenerator
import segcnn.utils as ut
import dvpy as dv
import dvpy.tf_2d
import segcnn
import glob
import U_Net_function_list as ff
import shutil

cg = segcnn.Experiment()


########### Define the model file########
batch = 0
Batch = str(batch)
epoch = '080'
view = '2C' # set as default
vector = '' # ignore
suffix = '' #ignore
test_set = 'VR_1tf_4class'
print(view,vector,Batch)

model_folder = os.path.join(cg.fc_dir,'models','model_batch'+Batch,'2D-UNet-seg')
filename = 'model-'+test_set+'_batch' + Batch + '_s' + suffix + '-' +epoch + '-*'
model_files = ff.find_all_target_files([filename],model_folder)
assert len(model_files) == 1
print(model_files)

seg_filename = 'pred_s_' #define file name of predicted segmentation
###########

dv.section_print('Loading Saved Weights...')

# BUILT U-NET

shape = cg.dim + (1,)
model_inputs = [Input(shape)]
model_outputs=[]
_, _, unet_output = dvpy.tf_2d.get_unet(cg.dim,
                                cg.num_classes,
                                cg.conv_depth,
                                layer_name='unet',
                                dimension =cg.unetdim,
                                unet_depth = cg.unet_depth,)(model_inputs[0])

model_outputs += [unet_output]
model = Model(inputs = model_inputs,outputs = model_outputs)
    
# Load weights
model.load_weights(model_files[0],by_name = True)
# build generator
valgen = dv.tf_2d.ImageDataGenerator(
                  cg.unetdim,
                  input_layer_names=['input_1'],
                  output_layer_names=['unet'],
                  )

#===========================================
dv.section_print('Get patient list...')
### Define the patient list you will predict on (MAY NEED TO write your own version)
# patient_list = ff.get_patient_list_from_csv(os.path.join(cg.spreadsheet_dir,'Final_patient_list_2020_after_Junes.csv'))
# print(len(patient_list))
patient_list = ff.find_all_target_files(['*/*'],cg.image_data_dir)
# patient_list = []
# for p in patient_list:
#   batch = ff.locate_batch_num_for_patient(p[0],p[1],os.path.join(cg.partition_dir,'partitions_lead_cases_local_adapted.npy'))
#   if batch == 1:
#     patient_list.append(p)
# print(patient_list)


#===========================================
dv.section_print('Prediction...')
count = 0
for p in patient_list:
  # patient_class = p[0]
  # patient_id = p[1]
  patient_class = os.path.basename(os.path.dirname(p))
  patient_id = os.path.basename(p)
  print(patient_class,patient_id)

  #############You can add a "continue" statement in the loop for cases you already done ###########

  # get a time frame list for that case
  l = ff.sort_timeframe(ff.find_all_target_files(['img-nii-0.625/*.nii.gz'],os.path.join(cg.image_data_dir,patient_class,patient_id)),2)
  time_frame_list = []
  for ll in l:
    time_frame_list.append(ff.find_timeframe(ll,2))
  print(time_frame_list)
 
  # define whether you have picked particular time frame(s) in the model training
  # for these time frames you don't need to make the prediction
  # for example you manually segmented time frame 0 and 0 was in the model training, t_picked = 0, pre_picked = 1
  # then you only need to predict frame 1-9 and directly copy your manual 0.

  if os.path.isfile(os.path.join(cg.seg_data_dir,patient_class,patient_id,'time_frame_picked_for_pretrained_AI.txt')) == 1:
    f = open(os.path.join(cg.seg_data_dir,patient_class,patient_id,'time_frame_picked_for_pretrained_AI.txt'),"r")
    t_picked = int(f.read())
    print(time_frame_list, ' picked: ', t_picked)
    pre_picked = 1
  else:
    print('did not pick time frame in the training')
    pre_picked = 0
    t_picked = 10000 # arbitray large number 


  ff.make_folder([os.path.join(cg.seg_data_dir,patient_class), os.path.join(cg.seg_data_dir,patient_class,patient_id)])

  # Prediction
  for t in time_frame_list:
    img = os.path.join(cg.local_dir,patient_class,patient_id,'img-nii-0.625',str(t)+'.nii.gz')
    assert os.path.isfile(img) == 1
    #seg = os.path.join(cg.local_dir,patient_class,patient_id,'seg-pred-1.5-upsample-retouch','pred_s_'+str(t_picked)+'.nii.gz')
    
    if os.path.isfile(os.path.join(cg.main_data_dir,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes',seg_filename + str(t) + '.nii.gz')) == 1:
      print('already done')
      continue

    #if t != t_picked:
    if 1 == 1:
      # define predict generator
      u_pred = model.predict_generator(valgen.predict_flow(np.asarray([img]),
                                                        slice_num = cg.slice_num,
                                                        batch_size = cg.slice_num,
                                                        relabel_LVOT = cg.relabel_LVOT,
                                                        shuffle = False,
                                                        input_adapter = ut.in_adapt,
                                                        output_adapter = ut.out_adapt,
                                                        shape = cg.dim,
                                                        input_channels = 1,
                                                        output_channels = cg.num_classes,
                                                        adapted_already = 0, 
                                                        ),
                                                        verbose = 1,
                                                        steps = 1,)

      # save u_net segmentation
      u_gt_nii = nb.load(os.path.join(cg.local_dir,patient_class,patient_id,'img-nii-0.625',str(t)+'.nii.gz')) # load image for affine matrix
      u_pred = np.rollaxis(u_pred, 0, 3)
      u_pred = np.argmax(u_pred , axis = -1).astype(np.uint8)
      u_pred = dv.crop_or_pad(u_pred, u_gt_nii.get_fdata().shape)
      u_pred[u_pred == 3] = 4  # particular for LVOT
      u_pred = nb.Nifti1Image(u_pred, u_gt_nii.affine)
      save_file = os.path.join(cg.main_data_dir,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes',seg_filename + str(t) + '.nii.gz')#predicted segmentation
      os.makedirs(os.path.dirname(save_file), exist_ok = True)
      nb.save(u_pred, save_file)



    #else: # already have manual segmentation in this time frame, only need to copy into new folder
      #assert t == t_picked
      #seg = os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-pred-1.5-upsample-retouch','pred_s_'+str(t_picked)+'.nii.gz')
      #shutil.copy(seg,os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-pred-0.625-4classes',seg_filename + str(t_picked) + '.nii.gz'))
 

 


