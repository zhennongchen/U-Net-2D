## parameters
# define GPU you use
export CUDA_VISIBLE_DEVICES="1"

# pixel size, for high resolution, set it to be 0.625
export CG_SPACING=0.625

# volume dimension: this should be set based on the dimension of every resampled volume
export CG_CROP_X=320 # has to be divisible by 2^5 = 32
export CG_CROP_Y=320 # has to be divisible by 2^5 = 32
export CG_CROP_Z=240

# define whether we are going to use pre-adapted image
export CG_ADAPTED_ALREADY=1

# set the batch:
# in one batch, the model will read n slices from N patients
# N should be divisible by both the total num of cases in training dataset as well as the total num of cases in validation dataset
export CG_PATIENTS_IN_ONE_BATCH=2  # set it to be larger will slow down the training speed.
# n should be divisible by CG_CROP_Z
# batch_size = N * n
export CG_BATCH_SIZE=24 # N = 2, n = 12


# set the number of classes in the output
export CG_NUM_CLASSES=4 #10 for Left-sided, 14 for Right-sided, 2 for LV only, 3 for LV+LA, 4 for LV+LA+LVOT (need relabel LVOT in the code)
export CG_RELABEL_LVOT=1

# set U-NET feature depth
export CG_CONV_DEPTH_MULTIPLIER=1
export CG_FEATURE_DEPTH=8 # 8 is up to 2^8 = 256, 9 is up to 512 and 10 is up to 1024, 2D-Unet should use 10

# set learning epochs
export CG_EPOCHS=60
export CG_LR_EPOCHS=26 # the number of epochs for learning rate change 

export CG_SEED=0

export CG_NUM_PARTITIONS=5

export CG_XY_RANGE="0.1"   #0.1

export CG_ZM_RANGE="0.1"  #0.1

export CG_RT_RANGE="10"   #15

export CG_NORMALIZE=0 #default = 0



## folders
export CG_PARENT_DIR="/Data/McVeighLabSuper/projects/Zhennong/AI/"

export CG_RAW_DIR="/Data/McVeighLabSuper/projects/Zhennong/AI/AI_datasets/"
export CG_BASE_DIR="/Data/McVeighLabSuper/projects/Zhennong/AI/CNN/" 


#export CG_OCTOMORE_DIR="/Experiment/Documents/Data"
export CG_OCTOMORE_DIR="/Data/local_storage/Zhennong/Data"
#export CG_OCTOMORE_DIR="/home/cnn/Documents/Data/"

export CG_FCNAS_DIR="/Data/ContijochLab/workspaces/zhennong/"

#####
export CG_INPUT_DIR="${CG_BASE_DIR}all-classes-all-phases-${CG_SPACING}/"

export CG_DERIVED_DIR="${CG_BASE_DIR}all-classes-all-phases-data-${CG_SPACING}/"

export CG_LOCAL_DIR="${CG_OCTOMORE_DIR}_${CG_SPACING}/"


