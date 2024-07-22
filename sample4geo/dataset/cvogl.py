import os
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import random
import copy
from tqdm import tqdm
import time
import albumentations
 

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

        self.train_file = os.path.join(data_folder, data_name, '{0}_{1}.pth'.format(data_name, 'train'))
        self.data_list = torch.load(self.train_file)

        train_ids_list = list()
        
        # for shuffle pool
        for data in self.data_list:
            idx = data[0]
            train_ids_list.append(idx)
            
        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)


    def __getitem__(self, index):
        # idx, sat, ground = self.data_list[index]
        idx, ground, sat, _, click_xy, gt_box, _, _ = self.data_list[index]

        # load query -> ground image
        query_img = cv2.imread(os.path.join(self.data_folder, self.data_name, 'query', ground))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(os.path.join(self.data_folder, self.data_name, 'satellite', sat))
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        scale_factor = 512.0/1024.0
        gt_box = [item * scale_factor for item in gt_box]

        click_xy = np.array(click_xy, dtype=int)
        gt_box = np.array(gt_box, dtype=int)

        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))

            # use roll for ground view if rotate sat view
            c, h, w = query_img.shape
            shifts = -w // 4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)

        label = torch.tensor(idx, dtype=torch.long)

        return query_img, reference_img, label, click_xy, np.array(gt_box, dtype=np.float32)

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
                
                if len(idx_pool) > 0:
                    idx = idx_pool.pop(0)

                    
                    if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:
                    
                        idx_batch.add(idx)
                        current_batch.append(idx)
                        idx_epoch.add(idx)
                        break_counter = 0
                      
                        if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                            
                            near_similarity = similarity_pool[idx][:neighbour_range]
                            
                            near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])
                            
                            far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])
                            
                            random.shuffle(far_neighbours)
                            
                            far_neighbours = far_neighbours[:neighbour_split]
                            
                            near_similarity_select = near_neighbours + far_neighbours
                            
                            for idx_near in near_similarity_select:
                           
                                # check for space in batch
                                if len(current_batch) >= self.shuffle_batch_size:
                                    break
                                
                                # check if idx not already in batch or epoch
                                if idx_near not in idx_batch and idx_near not in idx_epoch and idx_near:
                            
                                    idx_batch.add(idx_near)
                                    current_batch.append(idx_near)
                                    idx_epoch.add(idx_near)
                                    similarity_pool[idx].remove(idx_near)
                                    break_counter = 0
                                    
                    else:
                        # if idx fits not in batch and is not already used in epoch -> back to pool
                        if idx not in idx_batch and idx not in idx_epoch:
                            idx_pool.append(idx)
                            
                        break_counter += 1
                        
                    if break_counter >= 1024:
                        break
                   
                else:
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
        elif split == 'val':
            self.pth_file = f'{data_folder}/{data_name}/{data_name}_val.pth'
        else:
            self.pth_file = f'{data_folder}/{data_name}/{data_name}_test.pth'

        self.data_list = torch.load(self.pth_file)
        

    def __getitem__(self, index):
        idx, ground, sat, _, click_xy, gt_box, _, _ = self.data_list[index]

        # load query -> ground image
        query_img = cv2.imread(os.path.join(self.data_folder, self.data_name, 'query', ground))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(os.path.join(self.data_folder, self.data_name, 'satellite', sat))
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            reference_img = self.transforms[0](image=reference_img)['image']
            query_img = self.transforms[1](image=query_img)['image']

        scale_factor = 512.0/1024.0
        gt_box = [item * scale_factor for item in gt_box]

        click_xy = np.array(click_xy, dtype=int)
        gt_box = np.array(gt_box, dtype=int)

        label = torch.tensor(idx, dtype=torch.long)

        return query_img, reference_img, label, click_xy, np.array(gt_box, dtype=np.float32)

    def __len__(self):
        return len(self.data_list)
    


            





