# The code for SDM-VQA
# This code is for train-validation on LIVE, CSIQ and IVPL.
#
# Run with code `python3 demo_loop.py [id] [db]`, where [id] indicates the GPU number, and [db] denotes the database.
# 
# Created by Yongxu Liu (yongxu.liu@stu.xidian.edu.cn)

import sys
import os
from recorder import Recorder
from train_vqa import train_vqa
import numpy as np


# the number of reference videos
n_ref = {'LIVE': 10, 'CSIQ': 12, 'IVPL': 10}

n_param = len(sys.argv)
no_gpu = 0
db_name = 'LIVE'
if n_param > 1:  # GPU No.
    no_gpu = int(sys.argv[1])
if n_param > 2:  # db_name
    db_name = str(sys.argv[2])

os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % no_gpu

config_file = 'fr-vqa_' + db_name + '.yaml'
output_path = 'output_base_' + db_name + '_'
#output_path = 'test'
n_orgs = int(n_ref[db_name])
print(n_orgs)
if not os.path.exists(output_path):
    os.mkdir(output_path)

sys.stdout = Recorder(output_path + '/log.txt', sys.stdout)
# sys.stderr = Recorder(output_path + '/log.txt', sys.stderr)  # redirect std err, if necessary

srccs = []
# This only work in the condition that `n_ref` ranges from 8 to 12
# That is, 8 * 0.2 -> 2 <- 12 * 0.2
# for 8/2 splitting
for i in range(n_orgs-1):
    for j in range(i+1, n_orgs):   # for each possible instances
        best_srcc, best_epoch = \
            train_vqa(config_file=config_file, output_path=output_path,
                      is_random=False,
                      target=np.array([i, j]))

        srccs.append(best_srcc)
        with open(output_path + '/performance.txt', 'a') as f:
            f.write('[%d %d], srocc: %.4f, epoch: %d\n' % (i, j, best_srcc, best_epoch))

#
srccs = np.array(srccs)
med = np.median(srccs)
percent = np.percentile(srccs, [25, 50, 75])
print('performance: med-45: %.4f' % med)
print('Quartiles: %.4f - %.4f - %.4f ' % (percent[0], percent[1], percent[2]))
print('range: [%.4f - %.4f]' % (srccs.min(), srccs.max()))

with open(output_path + '/result.txt', 'a') as f:
    f.write('performance: med-45: %.4f\n' % med)
    f.write('Quartiles: %.4f - %.4f - %.4f\n' % (percent[0], percent[1], percent[2]))
    f.write('range: [%.4f - %.4f]\n' % (srccs.min(), srccs.max()))
    f.write('----------------------------\n\n')
