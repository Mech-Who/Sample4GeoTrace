import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        # print(f"{image_features1.shape=}") # [S, C]=[32, 1024]
        # print(f"{image_features2.shape=}") # [S, C]=[32, 1024]
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
    

class InfoNCESimilarityLoss(nn.Module):
    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.loss_fn = loss_function
        self.device = device

    def forward(self, query_feature, ref_feature, logit_scale, patch_size, click_points, gt_points):
        # 获取参考图像的目标点对应的图像块索引
        target_h_idx, target_w_idx = self.find_patch_index(gt_points, patch_size)

        # 提取图像中的所有块
        query_patches = self.extract_patches(query_feature, patch_size)
        target_patches = self.extract_patches(ref_feature, patch_size)

        # print(f"{query_patches.shape=}")
        # print(f"{target_patches.shape=}")

        # 找到查询图像的目标点所在的块的索引
        h_idx, w_idx = self.find_patch_index(click_points, patch_size)
        query_patch = query_patches[range(query_patches.shape[0]), :, h_idx, w_idx]
        # print(f"{query_patch.shape=}")

        # 计算相似度矩阵
        similarity_matrix = self.find_most_similar_patch(query_patch, target_patches).to(self.device)
        # print(f"{similarity_matrix.shape=}")
        # print(f"{query_patch.device=}")
        # print(f"{target_patches.device=}")
        # print(f"{similarity_matrix.device=}")
        # print(f"{logit_scale.device=}")

        # print("Most similar patch found.")
        # flatten the similarity matrix and the target indices
        batch_size, num_patches_h, num_patches_w = similarity_matrix.shape
        similarity_matrix = similarity_matrix.view(batch_size, -1)
        similarity_matrix = F.normalize(similarity_matrix, dim=-1)
        
        target_idx = target_h_idx * num_patches_w + target_w_idx
        # print(f"{target_idx.max()}")
        labels = target_idx.to(self.device)
        
        logits = (logit_scale * similarity_matrix).to(self.device)
        loss = self.loss_fn(logits, labels)

        return loss  
    
    def extract_patches(self, image, patch_size):
        # 提取图像中的所有块
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        return patches

    # 计算目标点所在的块的索引
    def find_patch_index(self, points, patch_size):
        
        # h_idx = click_points[:, 1] // patch_size
        # w_idx = click_points[:, 0] // patch_size
        h_idx = torch.div(points[:, 1], patch_size, rounding_mode='floor').long()
        w_idx = torch.div(points[:, 0], patch_size, rounding_mode='floor').long()
        return h_idx, w_idx

    # 计算两个块之间的相似性，这里使用余弦相似度
    def similarity(self, patch1, patch2):
        return F.cosine_similarity(patch1.reshape(patch1.shape[0], -1), patch2.reshape(patch2.shape[0], -1), dim=1)


    # 找到与查询块最相似的前k个块
    def find_most_similar_patch(self, query_patches, patches):
        batch_size, channels, num_patches_h, num_patches_w, patch_size, _ = patches.shape
        similarity_matrix = torch.zeros((batch_size, num_patches_h, num_patches_w))
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                target_patch = patches[:, :, i, j]
                sim = self.similarity(query_patches, target_patch)
                similarity_matrix[:, i, j] = sim

        return similarity_matrix
    
    def find_most_similar_k_patch(self, query_patches, patches, k=5):
        batch_size, channels, num_patches_h, num_patches_w, patch_size, _ = patches.shape
        similarity_matrix = torch.zeros((batch_size, num_patches_h, num_patches_w))
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                target_patch = patches[:, :, i, j]
                sim = self.similarity(query_patches, target_patch)
                similarity_matrix[:, i, j] = sim
        
        
        # 提取前k个最相似的图像块及其索引
        _, index = torch.topk(similarity_matrix.view(batch_size, -1), k, dim=-1, sorted=True)
        index_h = index // similarity_matrix.shape[2] # [batch_size, k]
        index_w = index % similarity_matrix.shape[2] # [batch_size, k]
        index_hw = index = torch.stack([index_h, index_w], dim=-1) # [batch_size, k, 2]
        top_k_patches = patches[torch.arange(similarity_matrix.shape[0])[:, None], :, index_h, index_w]
        
        return top_k_patches, index_hw


    


