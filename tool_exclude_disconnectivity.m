% this script can automatically exclude the disconnected parts in the
% segmentation and only keep the largest connectivity.
%%
close all;
clear all;
addpath(genpath('/Users/zhennongchen/Documents/GitHub/Volume_Rendering_by_DL/matlab/'));
%% List all patients
main_path = '/Volumes/Seagate MacOS/';
image_path = [main_path,'predicted_seg/Abnormal/'];
patient_list = Find_all_folders(image_path);
%% Do pixel clearning:
c = 0;
for i = 3:5
    patient_name = patient_list(i).name;
    disp(patient_name)
    patient_folder = [image_path,patient_name,'/seg-pred-0.625-4classes/'];
    if isfolder(patient_folder) == 1
       c = c+1;
       % make save_folder
       save_folder = [image_path,patient_name,'/seg-pred-0.625-4classes-connected-mat'];
       mkdir(save_folder)
      
       nii_list = Sort_time_frame(Find_all_files(patient_folder));
       for j = 1: size(nii_list,1)
           file_name = [patient_folder,convertStringsToChars(nii_list(j))];
           data = load_nii(file_name);
           image = data.img;
           
           % exclude
           BW = image > 0;
           [BW] = Find_largest_connected_component_3d(BW);
           % check
           CC = bwconncomp(BW);
           numPixels = cellfun(@numel,CC.PixelIdxList);
           if size(numPixels,2) > 1
               error('Error occurred in exclusion');
           end
           
           %%%%%%%%%%%%%%%%%%%%%
           % should have a better algorithm
           
           
           
           % put back to segmentation
           image(BW == 0) = 0;
           [image] = Transform_between_nii_and_mat_coordinate(image,1);
           % save
           t = Find_time_frame(convertStringsToChars(nii_list(j)),'_');
           save([save_folder,'/pred_s_',num2str(t),'.mat'],'image')
       end  
    else
        disp('not in the list') 
    end          
    end

        
    