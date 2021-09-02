# The code for SDM-VQA
# This code is for train-validation on IVC-IC.
# 
# Run with code `python3 demo_IVC-IC.py [id]`, where [id] indicates the GPU number.
# 
# Created by Yongxu Liu (yongxu.liu@stu.xidian.edu.cn)

import sys
import os
from recorder import Recorder
from train_vqa import train_vqa
import numpy as np

# for IVC-IC, we run cross-validation 20 times with generated random indices
n_ref = {'IVC-IC': 60}

n_param = len(sys.argv)
no_gpu = 0
db_name = 'IVC-IC'
if n_param > 1:  # GPU No.
    no_gpu = int(sys.argv[1])
# if n_param > 2:  # db_name
#     db_name = str(sys.argv[2])

os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % no_gpu

config_file = 'fr-vqa_' + db_name + '.yaml'
output_path = 'output_base_' + db_name + '_'
n_orgs = int(n_ref[db_name])
print(n_orgs)
if not os.path.exists(output_path):
    os.mkdir(output_path)

sys.stdout = Recorder(output_path + '/log.txt', sys.stdout)
# sys.stderr = Recorder(output_path + '/log.txt', sys.stderr)  # redirect std err, if necessary

srccs = []
for i in range(20):
    best_srcc, best_epoch = \
        train_vqa(config_file=config_file, output_path=output_path,
                  is_random=False,
                  target=np.array(range(i*2, i*2+12)))

    srccs.append(best_srcc)
    with open(output_path + '/performance.txt', 'a') as f:
        f.write('[%d], srocc: %.4f, epoch: %d\n' % (i, best_srcc, best_epoch))

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
