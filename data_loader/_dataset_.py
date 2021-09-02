from torch.utils.data import Dataset
import numpy as np
import random
from scipy.io import loadmat
import pickle
import os

class _dataset_(object):
    def __init__(self, db_config=None, target=None):
        """
        Initialize the data producer
        """
        assert db_config is not None

        self.db_name = db_config.get('db_name', 'LIVE')
        self.info_file = self.db_name + db_config.get('info_file', '_list_for_VQA.txt')
        self.data_path = db_config.get('data_path', '/mnt/disk/yongxu_liu/workspace/temporal_relation_diving/test_space_iFAST_kp/iFAST_key_10/')
        self.base_path = db_config.get('base_path', '/mnt/disk/yongxu_liu/datasets/') + self.db_name + '/'

        self.tr_te_r = float(db_config.get('train_size', 0.8))

        self.target = target
        self.is_random_split = False if target is not None else True

        self.train_dict = {}
        self.test_dict = {}

        self.read_info()

    def read_info(self):
        base = self.base_path
        ## read regular info
        ref_list, dis_list, d2r_list, score_list, width_list, height_list, fps_list = \
            [], [], [], [], [], [], []
        with open(self.info_file, 'r') as f:
            for line in f:
                scn_idx, dis_idx, ref, dis, score, width, height, fps = line.split()
                scn_idx = int(scn_idx)
                dis_idx = int(dis_idx)
                width = int(width)
                height = int(height)
                fps = int(fps)

                dis_list.append(base + dis)
                ref_list.append(base + ref)
                d2r_list.append(scn_idx)
                score_list.append(float(score))
                width_list.append(width)
                height_list.append(height)
                fps_list.append(fps)

        ref_list = np.asarray(ref_list)
        dis_list = np.asarray(dis_list)
        d2r_list = np.array(d2r_list, dtype='int')
        score_list = np.array(score_list, dtype='float32')
        width_list = np.array(width_list, dtype='int')
        height_list = np.array(height_list, dtype='int')
        fps_list = np.array(fps_list, dtype='int')

        # reverse DMOS -> MOS  |  NO
        score_list *= 0.8

        if os.path.exists(self.data_path + self.db_name + '.data.pkl'): 
            appearance, motion_content, motion_desc = self.load_prepared_data()
        else:
            appearance, motion_content, motion_desc = self.load_prepared_data_0(ref_list, dis_list)

        ## split
        scenes = np.unique(d2r_list)
        n_scenes = len(scenes)
        n_test = round(n_scenes * (1. - self.tr_te_r))

        if self.is_random_split:
            test_enum = random.sample(range(n_scenes), n_test)
            self.target = test_enum
        else:
            test_enum = self.target
        print('test_idx: ', test_enum)
        train_idx, test_idx, train_ref, test_ref = self.split_data(d2r_list, test_enum)
        print('train_: %d, test_: %d' % (len(train_idx), len(test_idx))) 
        self.train_dict = {'ref_list': ref_list[train_idx],
                           'dis_list': dis_list[train_idx],
                           'appearance': appearance[train_idx],
                           'motion_content': motion_content[train_idx],
                           'motion_desc': motion_desc[train_idx],
                           'd2r_list': self.re_arrange(d2r_list[train_idx]),
                           'score_list': score_list[train_idx],
                           'width_list': width_list[train_idx],
                           'height_list': height_list[train_idx],
                           'fps_list': fps_list[train_idx]}
        self.test_dict = {'ref_list': ref_list[test_idx],
                          'dis_list': dis_list[test_idx],
                          'appearance': appearance[test_idx],
                          'motion_content': motion_content[test_idx],
                          'motion_desc': motion_desc[test_idx],
                          'd2r_list': self.re_arrange(d2r_list[test_idx]),
                          'score_list': score_list[test_idx],
                          'width_list': width_list[test_idx],
                          'height_list': height_list[test_idx],
                          'fps_list': fps_list[test_idx]}

    def split_data(self, split_base, picked):

        n_data = len(split_base)

        train_idx, test_idx = [], []
        train_ref, test_ref = [], []
        for i in range(n_data):
            if split_base[i] in picked:
                test_idx.append(i)
            else:
                train_idx.append(i)
        for i in np.unique(split_base):
            if i in picked:
                test_ref.append(i)
            else:
                train_ref.append(i)
        return train_idx, test_idx, train_ref, test_ref

    def re_arrange(self, d2r_idx):
        n = len(d2r_idx)
        cnt = -1
        prev = -1
        for i in range(n):
            if d2r_idx[i] == prev:
                pass
            else:
                cnt += 1
                prev = d2r_idx[i]
            d2r_idx[i] = cnt

        return d2r_idx

    def load_prepared_data_0(self, ref_list, dis_list):
        # This function is to load from the original data created by MATLAB function.
        # Please check if files exits, or you need to create them with provided MATLAB function.
        appearance, motion_content, desc = [], [], []

        n_len = len(dis_list)
        for i in range(n_len):
            file = dis_list[i].split('/')[-1][:-4]
            data = loadmat(self.data_path + file + '.iFAST.mat')
            appearance.append(np.asarray(data['appearance']).astype('float32'))  # 10x18
            motion_content.append(np.asarray(data['content']).astype('float32').transpose((1, 0)))  # 10x4x8
 
            desc_orig = np.array(data['orig_desc']).astype('float32').transpose((3, 0, 1, 2))[:,:,np.newaxis,:,:]
            desc_dist = np.array(data['dist_desc']).astype('float32').transpose((3, 0, 1, 2))[:,:,np.newaxis,:,:]
            desc_simi = (2*desc_orig * desc_dist + 1e-6) / (desc_orig*desc_orig + desc_dist*desc_dist + 1e-6)
       
            desc.append(np.concatenate((desc_orig, desc_dist, desc_simi), axis=2))

        appearance = np.stack(appearance)
        motion_content = np.stack(motion_content)
        desc = np.stack(desc)
        
        print(appearance.shape)
        print(motion_content.shape)
        print(desc.shape)

        with open(self.data_path + self.db_name + '.data.pkl', 'wb') as f:
            pickle.dump((appearance, motion_content, desc), f)

        appearance /= 0.33
        motion_content /= 0.33
        return appearance, motion_content, desc

    def load_prepared_data(self):
        # This code is to load from preprocessed data.
        # The original data is created by MATLAB function. For a better usage in Python, the data is converted to `.pkl` files.
        with open(self.data_path + self.db_name + '.data.pkl', 'rb') as f:
            appearance, motion_content, desc = pickle.load(f)

        appearance /= 0.33
        motion_content /= 0.33
        return appearance, motion_content, desc


class Dataset_this(Dataset):

    def __init__(self, dict=None, is_train=True, is_shuffle=False):
        self.info_dict = dict
        self.n_data = len(dict['dis_list'])
        self.is_train = is_train
        self.is_shuffle = is_shuffle

        self.d2r_data = []
        self.mos = []

        self.score = dict['score_list']
        self.d2r = dict['d2r_list']
        self.fps = dict['fps_list']

        self.appearance = dict['appearance']
        self.motion_content = dict['motion_content']
        self.motion_desc = dict['motion_desc']

        self.d2r_data = np.asarray(self.d2r_data)
        self.mos = np.asarray(self.mos)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        appearance = self.appearance[index]
        motion_content = self.motion_content[index]
        motion_desc = self.motion_desc[index]

        mos = self.score[index]

        return (appearance, motion_content, motion_desc, mos)

    def __len__(self):
        return self.n_data
