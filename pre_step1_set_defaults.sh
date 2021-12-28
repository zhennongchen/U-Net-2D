## parameters
# define GPU you use
export CUDA_VISIBLE_DEVICES="0"

# pixel size, for high resolution, set it to be 0.625, low res = 1.5
export CG_SPACING=0.625

# volume dimension: this should be set based on the dimension of every resampled volume
# use tool_check_image_size.py to check the dimension
export CG_CROP_X=352 # has to be divisible by 2^5 = 32
export CG_CROP_Y=352 # has to be divisible by 2^5 = 32
export CG_CROP_Z=256 # has to be divisible by 2^5 = 32

# define whether we are going to use pre-adapted image (turn it on to facilitate the training)
export CG_ADAPTED_ALREADY=1

# set the batch:
# in one batch, the model will read n slices from N patients
# N should be divisible by both the total num of cases in training dataset as well as the total num of cases in validation dataset
export CG_PATIENTS_IN_ONE_BATCH=2  # set it to be larger will slow down the training speed.
# n should be divisible by CG_CROP_Z
# batch_size = N * n
export CG_BATCH_SIZE=32 # N = 2 patients in one batch, n = 16 slices from each patient


# set the number of classes in the output
export CG_NUM_CLASSES=4 #10 for Left-sided, 14 for Right-sided, 2 for LV only, 3 for LV+LA, 4 for LV+LA+LVOT (need relabel LVOT in the code)
export CG_RELABEL_LVOT=1

# set U-NET feature depth (no need to change)
export CG_CONV_DEPTH_MULTIPLIER=1 # default = 1 
export CG_FEATURE_DEPTH=8 # 8 is up to 2^8 = 256, 9 is up to 512 and 10 is up to 1024, 2D-Unet should use 10

# set learning epochs
export CG_EPOCHS=80
export CG_LR_EPOCHS=26 # the number of epochs for learning rate change 

export CG_SEED=1

export CG_NUM_PARTITIONS=5

export CG_XY_RANGE="0.1"   #0.1

export CG_ZM_RANGE="0.1"  #0.1

export CG_RT_RANGE="10"   #15

export CG_NORMALIZE=0 #default = 0



# folders for Zhennong's dataset
export CG_MAIN_DATA_DIR="/Data/McVeighLabSuper/wip/Ashish_ResyncCT/"   # main folder in NAS
export CG_IMAGE_DATA_DIR="${CG_MAIN_DATA_DIR}nii-images/"       # folder in NAS to save all nii images (raw image, resampled image, adapted image)
export CG_SEG_DATA_DIR="${CG_MAIN_DATA_DIR}predicted_seg/"      # folder in NAS to save all segmentations (manual ones, predicted ones)
export CG_SPREADSHEET_DIR="/Data/McVeighLabSuper/wip/zhennong/spreadsheets/"    # folder in NAS to save all spreadsheets
export CG_PARTITION_DIR="${CG_MAIN_DATA_DIR}partition/"         # folder in NAS to save partition files
export CG_OCTOMORE_DIR="/Data/local_storage/Zhennong/Ashish_ResyncCT/"   # folder in octomore local_storage to save all image files
export CG_FCNAS_DIR="/Data/ContijochLab/workspaces/zhennong/Volume_Rendering_segmentation/"  # folder in NAS (francisco's NAS) to save all files for AI model weights



