## parameters
export CUDA_VISIBLE_DEVICES="1"

export CG_NUM_CLASSES=4 #10 for Left-sided, 14 for Right-sided, 2 for LV only, 3 for LV+LA, 4 for LV+LA+LVOT (need relabel LVOT in the code)
export CG_RELABEL_LVOT=1

export CG_NORMALIZE=0 #default = 0

export CG_SPACING=1.5

export CG_FEATURE_DEPTH=8
# 8 is up to 2^8 = 256, 9 is up to 512 and 10 is up to 1024

export CG_EPOCHS=40

export CG_SEED=0

export CG_LR_EPOCHS=26

export CG_NUM_PARTITIONS=5

export CG_BATCH_SIZE=1

export CG_XY_RANGE="0.1"   #0.1

export CG_ZM_RANGE="0.1"  #0.1

export CG_RT_RANGE="10"   #15

export CG_CROP_X=160
export CG_CROP_Y=160
export CG_CROP_Z=96

export CG_CONV_DEPTH_MULTIPLIER=1


## folders
export CG_PARENT_DIR="/Data/McVeighLabSuper/projects/Zhennong/AI/"

export CG_RAW_DIR="/Data/McVeighLabSuper/projects/Zhennong/AI/AI_datasets/"
export CG_BASE_DIR="/Data/McVeighLabSuper/projects/Zhennong/AI/CNN/" 
#export CG_RAW_DIR="/media/McVeighLabSuper/projects/Zhennong/AI/AI_datasets/"
#export CG_BASE_DIR="/media/McVeighLabSuper/projects/Zhennong/AI/CNN/" 

export CG_CODE_DIR="/Experiment/Documents/AI_reslice_orthogonal_view/"

export CG_HYPERTUNE_DIR="${CG_BASE_DIR_ZC}hyperparameters_tuning/"

export CG_OCTOMORE_DIR="/Experiment/Documents/Data"
#export CG_OCTOMORE_DIR="/home/cnn/Documents/Data/"

export CG_FCNAS_DIR="/Data/ContijochLab/workspaces/zhennong/"


