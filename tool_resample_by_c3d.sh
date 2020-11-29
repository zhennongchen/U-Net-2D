#!/usr/bin/env bash
# run in docker c3d

half=2
minus=1

# Include settings and ${CG_*} variables.
. defaults.sh

# Get a list of patients.
patients=(/Data/McVeighLabSuper/projects/Zhennong/AI/AI_datasets/*/*)
save_folder="/Data/McVeighLabSuper/projects/Zhennong/AI/CNN/all-classes-all-phases-0.625/"
img_or_seg=0 # 1 is image, 0 is seg

if ((${img_or_seg} == 1))
then
img_folder="img-nii"
else
img_folder="seg-nii"
fi

for p in ${patients[*]};
do

# Print the current patient.
  echo ${p} 
  
  # assert whether dcm image exists
  if ! [ -d ${p}/${img_folder} ] || ! [ "$(ls -A  ${p}/${img_folder})" ];then
    echo "no image/seg"
    continue
  fi

  # find out patientclass and patientid
  patient_id=$(basename ${p})
  patient_class=$(basename $(dirname ${p}))

  # set output folder
  
  if ((${img_or_seg} == 1))
  then
  o_dir=${save_folder}${patient_class}/${patient_id}/img-nii-0.625
  else
  o_dir=${save_folder}${patient_class}/${patient_id}/seg-nii-0.625
  fi

  echo ${o_dir}
  mkdir -p ${o_dir}
  
  IMGS=(${p}/${img_folder}/*.nii.gz)

  for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
  do
  #echo ${IMGS[${i}]}
    i_file=${IMGS[${i}]}
    echo ${i_file}
    o_file=${o_dir}/$(basename ${i_file})

    if [ -f ${o_file} ];then
      echo "already done this file"
      continue
    else
      if ((${img_or_seg} == 1))
      then
        c3d ${i_file} -interpolation Cubic -resample-mm 0.625x0.625x0.625mm -o ${o_file}
      else
        c3d ${i_file} -interpolation NearestNeighbor -resample-mm 0.625mmx0.625mmx0.625mm -o ${o_file}
      fi
    fi   
  done
done


