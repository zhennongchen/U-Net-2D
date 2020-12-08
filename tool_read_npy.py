#!/usr/bin/env python
import numpy as np
import os
import nibabel as nb
import math
import segcnn
import zc_function_list as ff
#np.set_printoptions(precision=3,suppress=True)

cg = segcnn.Experiment()

data = np.load(os.path.join(cg.data_dir,'ED_ES_adapted','img_list_0.npy'))
print(data)


