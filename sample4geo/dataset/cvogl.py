import os
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import random
import copy
from tqdm import tqdm
import time

 

class CVOGLDatasetTrain(Dataset):
    def __init__(self,
                 data_folder,
                 data_name,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128):
        super().__init__()
        self.data_folder = data_folder
        self.data_name = data_name
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference

        self.train_file = os.path.join(data_folder, data_name, f'{data_name}_train.pth')
        self.data_list = torch.load(self.train_file)

        self.idx2pair = dict()
        train_ids_list = list()
        
        # for shuffle pool
        for data in self.data_list:
            idx, query_img_name, ref_img_name, _, click_pt, bbox, _, _ = data
            self.idx2pair[idx] = (idx, query_img_name, ref_img_name, click_pt, bbox)
            train_ids_list.append(idx)

        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)


    def __getitem__(self, index):
        # idx, sat, ground = self.data_list[index]
        idx, ground, sat, click_xy, gt_box = self.idx2pair[self.samples[index]]

        # load query -> ground image
        query_img = cv2.imread(os.path.join(self.data_folder, self.data_name, 'query', ground))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(os.path.join(self.data_folder, self.data_name, 'satellite', sat))
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        click_xy = np.array(click_xy, dtype=int)

        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            # TODO：翻转时， click_xy 需要一起进行翻转
            reference_img = cv2.flip(reference_img, 1)
            # TODO：翻转时， gt_box 需要一起进行翻转

        # image transforms
        # TODO：缩放变换需要把click和bbox一起进行修改
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))
            # TODO： gt_box 需要和 ref 图像一起旋转
            # use roll for ground view if rotate sat view
            c, h, w = query_img.shape
            shifts = -w // 4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)
            # TODO： click_xy 需要和 query 图像一起旋转

        label = torch.tensor(idx, dtype=torch.long)

        # TODO：是否需要返回 click_xy 和 gt_box 这两个？
        return query_img, reference_img, label

    def __len__(self):
        return len(self.samples)
        
    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            idx_pool = copy.deepcopy(self.train_ids)
        
            neighbour_split = neighbour_select // 2
            
            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)
            else:
                similarity_pool = None
                
            # Shuffle pairs order
            random.shuffle(idx_pool)
           
            # Lookup if already used in epoch
            idx_epoch = set()   
            idx_batch = set()
     
            # buckets
            batches = []
            current_batch = []
            
            # counter
            break_counter = 0
            
            # progressbar
            pbar = tqdm()
    
            while True:
                pbar.update()
                if len(idx_pool) <= 0:
                    break
                idx = idx_pool.pop(0)
                # 幂等处理
                # 1. 既没有在当前batch中用过
                # 2. 也没有在之前的epoch中用过
                # 3. 并且当前批次没有遍历完
                if (idx not in idx_batch) and (idx not in idx_epoch) and (len(current_batch) < self.shuffle_batch_size):
                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)

                    break_counter = 0
                    # 1. similarity_pool 非空
                    # 2. 当前批次未结束
                    if (similarity_pool is not None) and (len(current_batch) < self.shuffle_batch_size):
                        # 取最相似的 neighbour_range 个图像
                        near_similarity = similarity_pool[idx][:neighbour_range]
                        # 分别记录前一部分（near）和后一部分（far）（例如默认是0-32，32-128）
                        near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])
                        far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])
                        # 打乱后一半的顺序
                        random.shuffle(far_neighbours)
                        # 取后面的部分中的前 neighbour_split 个记录（默认是32）
                        far_neighbours = far_neighbours[:neighbour_split]
                        # 合并前后两个部分，共 2*neighbour_split 个记录（默认64）
                        near_similarity_select = near_neighbours + far_neighbours
                        for idx_near in near_similarity_select:
                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break
                            # （类似于做幂等）check if idx not already in batch or epoch
                            if idx_near not in idx_batch and idx_near not in idx_epoch and idx_near:
                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0
                else:
                    # if idx fits not in batch and is not already used in epoch -> back to pool
                    # 如果是 if 判断的前两种情况，则添加 idx 到 idx_pool
                    if idx not in idx_batch and idx not in idx_epoch:
                        idx_pool.append(idx)
                    break_counter += 1
                if break_counter >= 1024:
                    break
                if len(current_batch) >= self.shuffle_batch_size:
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []

            pbar.close()
            # wait before closing progress bar
            time.sleep(0.3)
            self.samples = batches
            print("idx_pool:", len(idx_pool))
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_ids), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.train_ids) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))  


class CVOGLDatasetEval(Dataset):
    def __init__(self,
                 data_folder,
                 data_name,
                 split,
                 img_type,
                 transforms=None,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.data_name = data_name
        self.split = split
        self.img_type = img_type
        self.transforms = transforms

        if split == 'train':
            self.pth_file = f'{data_folder}/{data_name}/{data_name}_train.pth'
        else:
            self.pth_file = f'{data_folder}/{data_name}/{data_name}_test.pth'

        self.data_list = torch.load(self.pth_file)

        self.idx2sat = {}
        self.idx2ground = {}
        self.click_xy = []
        self._load_data()

        

    def _load_data(self):
        for data in self.data_list:
            idx = data[0]
            ground_image = data[1]
            sat_image = data[2]
            self.click_xy.append(data[4])
            self.idx2sat[idx] = sat_image
            self.idx2ground[idx] = ground_image

        if self.img_type == "reference":
            self.images = list(self.idx2sat.values())
            self.label = list(self.idx2sat.keys())
        elif self.img_type == "query":
            self.images = list(self.idx2ground.values())
            self.label = list(self.idx2ground.keys())
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
        

    def __getitem__(self, index):
        
        if self.img_type == "reference":
            img = cv2.imread(f'{self.data_folder}/{self.data_name}/satellite/{self.images[index]}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.img_type == "query":
            img = cv2.imread(f'{self.data_folder}/{self.data_name}/query/{self.images[index]}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
        
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)

            





