#!/usr/bin/env python

# in terminal, type ./zc_04_train --batch No. to run

# System
import argparse
import os

# Third Party
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import CSVLogger
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Input, \
                         Conv1D, Conv2D, Conv3D, \
                         MaxPooling1D, MaxPooling2D, MaxPooling3D, \
                         UpSampling1D, UpSampling2D, UpSampling3D, \
                         Reshape, Flatten, Dense
from keras.layers.merge import concatenate, multiply
from keras.initializers import Orthogonal
from keras.regularizers import l2
from keras.layers.merge import concatenate, multiply

import tensorflow as tf

# Internal
import segcnn.utils as ut
import dvpy as dv
import dvpy.tf_2d
import segcnn

cg = segcnn.Experiment()
fs = segcnn.FileSystem(cg.base_dir, cg.data_dir,cg.local_dir)

K.set_image_dim_ordering('tf')  # Tensorflow dimension ordering in this code

# Allow Dynamic memory allocation.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def train(batch):
    print(cg.dim)
    print('BATCH_SIZE = ',cg.batch_size)
    
    # define test_set
    test_set = '2D'

    # define views and vector
    view = '2C'

    # define partition file
    partition_file_name = 'ED_ES_U2'

    # define hdf5 file save folder
    weight_file_save_folder = os.path.join(cg.fc_dir,'models')
    #===========================================
    dv.section_print('Calculating Image Lists...')

    imgs_list_trn=[np.load(fs.img_list(p, partition_file_name)) for p in range(cg.num_partitions)]
    segs_list_trn=[np.load(fs.seg_list(p, partition_file_name)) for p in range(cg.num_partitions)]
   

    imgs_list_tst = imgs_list_trn.pop(batch)
    segs_list_tst = segs_list_trn.pop(batch)
      

    imgs_list_trn = np.concatenate(imgs_list_trn)
    segs_list_trn = np.concatenate(segs_list_trn)

    len_list=[len(imgs_list_trn),len(segs_list_trn),len(imgs_list_tst),len(segs_list_tst)]
    print(len_list)

    #===========================================
    dv.section_print('Creating and compiling model...')
    shape = cg.dim + (1,)
    model_inputs = [Input(shape)]
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
    opt = Adam(lr = 1e-4) 
    losses={'unet':'categorical_crossentropy'} 
    model.compile(optimizer= opt, 
                 loss= losses,
                 metrics= {'unet':'acc',})
    
    #======================
    dv.section_print('Fitting model...')
   
    if batch is None:
      model_name = 'model-'+test_set+'_batchall_s'
      model_fld = 'model_batchall'
    else:
      model_name = 'model-'+test_set+'_batch'+str(batch)+'_s'
      model_fld = 'model_batch'+str(batch)
    filename = model_name +'-{epoch:03d}-{loss:.3f}-{val_loss:.3f}-{val_acc:.4f}.hdf5'
    filepath=os.path.join(weight_file_save_folder,model_fld,'2D-UNet',filename)   
    os.makedirs(os.path.dirname(filepath), exist_ok = True)  
  
    # set callbacks
    #csv_logger = CSVLogger(os.path.join(cg.fc_dir, 'logs',  model_name + '_training-log' + '.csv'))
    callbacks = [#csv_logger,
                 ModelCheckpoint(filepath,          
                                 monitor='val_loss',
                                 save_best_only=False,
                                 ),
                 LearningRateScheduler(dv.learning_rate_step_decay),   
                ]
   

    datagen = dv.tf_2d.ImageDataGenerator(
        cg.unetdim,  # Dimension of input image
        input_layer_names = ['input_1'],
        output_layer_names = ['unet'],
        translation_range=cg.xy_range,  # randomly shift images vertically (fraction of total height)
        rotation_range=cg.rt_range,  # randomly rotate images in the range (degrees, 0 to 180)
        scale_range=cg.zm_range,
        flip=cg.flip,)
    
    datagen_flow = datagen.flow(imgs_list_trn,
      segs_list_trn,
      batch_size = cg.batch_size,
      slice_num = 96,
      view = view,
      relabel_LVOT = cg.relabel_LVOT,
      input_adapter = ut.in_adapt,
      output_adapter = ut.out_adapt,
      shape = cg.dim,
      input_channels = 1,
      output_channels = cg.num_classes,
      augment = True,
      normalize = cg.normalize,
      )

    valgen = dv.tf_2d.ImageDataGenerator(
        cg.unetdim, 
        input_layer_names=['input_1'],
        output_layer_names=['unet'],
        )

    valgen_flow = valgen.flow(imgs_list_tst,
      segs_list_tst,
      batch_size = cg.batch_size,
      slice_num = 96,
      view = view,
      relabel_LVOT = cg.relabel_LVOT,
      input_adapter = ut.in_adapt,
      output_adapter = ut.out_adapt,
      shape = cg.dim,
      input_channels = 1,
      output_channels = cg.num_classes,
      normalize = cg.normalize,
      )

    # Fit the model on the batches generated by datagen.flow().
  
    model.fit_generator(datagen_flow,
                        steps_per_epoch = imgs_list_trn.shape[0] * 96 // cg.batch_size,
                        epochs = cg.epochs,
                        workers = 1,
                        validation_data = valgen_flow,
                        validation_steps = imgs_list_tst.shape[0] * 96 // cg.batch_size,
                        callbacks = callbacks,
                        verbose = 1,
                       )

    
if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int)
  args = parser.parse_args()

  if args.batch is not None:
    assert(0 <= args.batch < cg.num_partitions)

  train(args.batch)
  