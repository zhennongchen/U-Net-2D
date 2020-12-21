#!/usr/bin/env python
import numpy as np
import os
import nibabel as nb
import math
import segcnn
import U_Net_function_list as ff
#np.set_printoptions(precision=3,suppress=True)

cg = segcnn.Experiment()

data = np.load(os.path.join(cg.partition_dir,'partitions_local_adapted.npy'),allow_pickle = True)
print(data)




