#!/usr/bin/env python

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
import zc_function_list as ff

cg = segcnn.Experiment()
fs = segcnn.FileSystem(cg.base_dir, cg.data_dir,cg.local_dir)


###########
Batch = '0'
segmentation_pred = True
epoch = '043'
view = '2C'
vector = ''
suffix = '' #sometime we have r2 or t3.
test_set = '2D'
print(view,vector,Batch)

model_folder = os.path.join(cg.fc_dir,'models','model_batch'+Batch,'2D-UNet')
#model_folder = os.path.join(cg.data_dir,'model_batch'+Batch)

if segmentation_pred == True:
  filename = 'model-'+test_set+'_batch' + Batch + '_s' + suffix + '-' +epoch + '-*'
else:
  filename = 'model-'+test_set+'_batch' + Batch + '_' + view + '_' + vector + suffix + '-' + epoch +'-*'

model_files = ff.find_all_target_files([filename],model_folder)
assert len(model_files) == 1
print(model_files)

matrix_filename = test_set+'_'+view+'_'+vector
seg_filename = 'UNet_'+test_set+'_s_'
###########


def predict(batch):

    #===========================================
    dv.section_print('Calculating Image Lists...')

    imgs_list_tst=[np.load(fs.img_list(p, 'ED_ES_U2')) for p in range(cg.num_partitions)]
    segs_list_tst=[np.load(fs.seg_list(p, 'ED_ES_U2')) for p in range(cg.num_partitions)]
    
    if batch is None:
      raise ValueError('No batch was provided: wrong!')
      batch = 'all'
      imgs_list_tst = np.concatenate(imgs_list_tst)
      segs_list_tst = np.concatenate(segs_list_tst)
    else:
      imgs_list_tst = imgs_list_tst[batch]
      segs_list_tst = segs_list_tst[batch]
      
    print(imgs_list_tst.shape)
    #===========================================
    dv.section_print('Loading Saved Weights...')

    # Input size is unknown
    shape = cg.dim + (1,)

    model_inputs = [Input(shape)]

    # Input size is unknown
    model_outputs=[]
    _, _, unet_output = dvpy.tf_2d.get_unet(cg.dim,
                                    cg.num_classes,
                                    cg.conv_depth,
                                    layer_name='unet',
                                    dimension =cg.unetdim,
                                    unet_depth = cg.unet_depth,
                                   )(model_inputs[0])
    model_outputs += [unet_output]

    model = Model(inputs = model_inputs,outputs = model_outputs)
    
    # Load weights
    model.load_weights(model_files[0],by_name = True)
  
    #===========================================
    dv.section_print('Calculating Predictions...')

    valgen = dv.tf_2d.ImageDataGenerator(
                  cg.unetdim,
                  input_layer_names=['input_1'],
                  output_layer_names=['unet'],
                  )

    # predict
    if segmentation_pred == False:
      class_list = []
      id_list = []
      matrix_list = []

    for img, seg in zip(imgs_list_tst, segs_list_tst):
      patient_id = os.path.basename(os.path.dirname(os.path.dirname(img)))
      patient_class = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img))))
      print(patient_class,patient_id,'\n')

      u_pred = model.predict_generator(valgen.flow(np.asarray([img]),np.asarray([seg]),
      batch_size = 96,
      slice_num = 96,
      view = view,
      relabel_LVOT = cg.relabel_LVOT,
      shuffle = False,
      input_adapter = ut.in_adapt,
      output_adapter = ut.out_adapt,
      shape = cg.dim,
      input_channels = 1,
      output_channels = cg.num_classes,
      ),
      verbose = 1,
      steps = 1,)
                                          
      #save u_net segmentation
      if segmentation_pred == True:
        u_gt_nii = nb.load(seg)
        u_pred = np.rollaxis(u_pred, 0, 3)
        u_pred = np.argmax(u_pred , axis = -1).astype(np.uint8)
        u_pred = dv.crop_or_pad(u_pred, u_gt_nii.get_fdata().shape)
        u_pred[u_pred == 3] = 4
        u_pred = nb.Nifti1Image(u_pred, u_gt_nii.affine)
        save_file = os.path.join(cg.base_dir,patient_class,patient_id,cg.pred_dir,seg_filename + os.path.basename(seg))
        os.makedirs(os.path.dirname(save_file), exist_ok = True)
        nb.save(u_pred, save_file)
      else:
        # save matrix
        x_n = ff.normalize(x_pred)
        y_n = ff.normalize(y_pred)
        matrix = np.concatenate((t_pred.reshape(1,3),x_n.reshape(1,3),y_n.reshape(1,3)))
        class_list.append(patient_class)
        id_list.append(patient_id)
        matrix_list.append(matrix)

  # select best prediction of vectors from ED and ES 
    if segmentation_pred == False:
      #assert len(matrix_list) == len(imgs_list_tst)
      for i in range(0,len(matrix_list)):
        if i%2 == 1:
          continue
        else:
          # load ground truth
          gt = np.load(os.path.join(cg.base_dir,class_list[i],id_list[i],'affine_standard',view+'_MR.npy'),allow_pickle = True)
          [t_gt,x_gt,y_gt] = [gt[12],gt[5],gt[7]]
          # calculate and compare the error
          # if translation:
          if vector == 't':
            error1 = math.sqrt((mean_squared_error(ff.turn_to_pixel(matrix_list[i][0]),t_gt)) * 3) * 1.5
            error2 = math.sqrt((mean_squared_error(ff.turn_to_pixel(matrix_list[i+1][0]),t_gt)) * 3) * 1.5
          # if direction:
          else: 
            error1 = ff.orientation_error(x_gt,y_gt,matrix_list[i][1],matrix_list[i][-1])
            error2 = ff.orientation_error(x_gt,y_gt,matrix_list[i+1][1],matrix_list[i+1][-1])
          # save matrix
          save_file = os.path.join(cg.base_dir,class_list[i],id_list[i],'matrix-pred',matrix_filename)
          os.makedirs(os.path.dirname(save_file),exist_ok = True)
          if error1 <= error2:
            print(class_list[i],id_list[i],' ED best because ED error is ', round(error1,2),' and ES is ', round(error2,2))
            np.save(save_file,matrix_list[i])
          else:
            print(class_list[i],id_list[i],' ES best because ED error is ', round(error1,2),' and ES is ', round(error2,2))
            np.save(save_file,matrix_list[i+1])
      
      
if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int)
  args = parser.parse_args()

  if args.batch is not None:
    assert(0 <= args.batch < cg.num_partitions)

  predict(args.batch)