#!/usr/bin/env bash
# run in docker c3d

half=2
minus=1

# Include settings and ${CG_*} variables.
# . defaults.sh

# Get a list of patients.
patients=(/Data/McVeighLabSuper/wip/Ashish_ResyncCT/nii-images/*/*)
save_folder="/Data/McVeighLabSuper/wip/Ashish_ResyncCT/nii-images/"
img_or_seg=1 # 1 is image, 0 is seg

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
  o_dir=${save_folder}${patient_class}/${patient_id}/img-nii-1.5
  else
  o_dir=${save_folder}${patient_class}/${patient_id}/seg-nii-0.625
  fi

  echo ${o_dir}
  mkdir -p ${o_dir}

  # read the es.txt file
  # es_file=$(<${p}/es.txt)

  IMGS=(${p}/${img_folder}/*.nii.gz)
  #IMGS+=(${p}/${img_folder}/${es_file}.nii.gz)


  for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
  do
    i_file=${IMGS[${i}]}
    echo ${i_file}
    o_file=${o_dir}/$(basename ${i_file})

    if [ -f ${o_file} ];then
      echo "already done this file"
      continue
    else
      if ((${img_or_seg} == 1))
      then
        c3d ${i_file} -interpolation Cubic -resample-mm 1.5x1.5x1.5mm -o ${o_file}
      else
        c3d ${i_file} -interpolation NearestNeighbor -resample-mm 0.625mmx0.625mmx0.625mm -o ${o_file}
      fi
    fi   
  done
done


