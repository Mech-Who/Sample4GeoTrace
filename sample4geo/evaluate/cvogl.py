import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
import torch.nn.functional as F
from ..trainer import predict
from ..utils import xyxy2xywh, xywh2xyxy, bbox_iou


def evaluate(config,
             model,
             dataloader,
             cleanup=True):

    print("\nExtract Features:")
    query_features, ref_features, scale1, scale2, click_xy, gt_box = predict(
        config, model, dataloader)

    print("Compute Scores:")
    score = calculate_scores(config, query_features,
                             ref_features, scale1, scale2, click_xy, gt_box)

    # cleanup and free memory on GPU
    if cleanup:
        del query_features, ref_features, scale1, scale2, click_xy, gt_box
        gc.collect()

    return score


def calc_sim(config,
             model,
             reference_dataloader,
             query_dataloader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):

    print("\nExtract Features:")
    reference_features, reference_labels = predict(
        config, model, reference_dataloader)
    query_features, query_labels = predict(config, model, query_dataloader)

    print("Compute Scores Train:")
    r1 = calculate_scores(query_features, reference_features, query_labels,
                          reference_labels, step_size=step_size, ranks=ranks)

    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)

    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()

    return r1, near_dict


def calculate_scores(config, query_feature, ref_feature, scale1, scale2, click_xy, gt_box):
    batch_size = gt_box.shape[0]
    patch_size = int(config.patch_size // scale2)
    click_points = click_xy // scale1
    gt_box = xyxy2xywh(gt_box)
    # print(f"{gt_box.shape=}")
    # gt_points = (gt_box[:, 0:2] / scale2).long()

    # 提取图像中的所有块
    query_patches = extract_patches(query_feature, patch_size)
    target_patches = extract_patches(ref_feature, patch_size)

    # print(f"{query_patches.shape=}")
    # print(f"{target_patches.shape=}")

    # 找到查询图像的目标点所在的块的索引
    h_idx, w_idx = find_patch_index(click_points, patch_size)
    # print(f"{query_patches.shape=}")
    # print(f"{h_idx.max()=},{w_idx.max()=}")
    query_patch = query_patches[range(query_patches.shape[0]), :, h_idx, w_idx]

    # 计算相似度矩阵，获取前五个最相似的图像块
    # [batch_size, k, channels, patch_size, patch_size], [batch_size, k, 2]
    top_k_patches, index_hw = find_most_similar_k_patch(
        query_patch, target_patches, 5)

    # 将图像块转换成bbox
    bbox = convert_patches_to_bbox(top_k_patches, index_hw).to(config.device)
    # print(f"{bbox.shape=}")
    # print(f"{gt_box.device=}, {bbox.device=}")

    # 还原成真正的预测框
    bbox = bbox * scale2
    bbox = torch.clamp(bbox, min=0, max=config.img_size-1)

    # 计算得分
    _, k, _ = bbox.shape
    iou = bbox_iou(gt_box.repeat(1, k).reshape(-1, 4),
                   bbox.reshape(-1, 4), x1y1x2y2=False)
    iou = iou.reshape(batch_size, k)

    accu_list, threshold_list = list(), [0.5, 0.25]
    for threshold in threshold_list:
        # 判断是否至少有一个bbox的IOU大于threshold
        is_correct = (iou.max(dim=1)[0] > threshold).float()
        accu = is_correct.mean().item()
        accu_list.append(accu)

    return accu_list


def extract_patches(image, patch_size):
    # 提取图像中的所有块
    patches = image.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size)
    return patches


# 计算目标点所在的块的索引
def find_patch_index(points, patch_size):

    # h_idx = click_points[:, 1] // patch_size
    # w_idx = click_points[:, 0] // patch_size
    h_idx = torch.div(points[:, 1], patch_size, rounding_mode='floor').long()
    w_idx = torch.div(points[:, 0], patch_size, rounding_mode='floor').long()
    return h_idx, w_idx


# 计算两个块之间的相似性，这里使用余弦相似度
def similarity(patch1, patch2):
    sim = F.cosine_similarity(patch1.reshape(
        patch1.shape[0], -1), patch2.reshape(patch2.shape[0], -1), dim=1)
    sim = torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    return sim


# 找到与查询块最相似的块
def find_most_similar_k_patch(query_patches, patches, k=5):
    batch_size, channels, num_patches_h, num_patches_w, patch_size, _ = patches.shape
    similarity_matrix = torch.zeros((batch_size, num_patches_h, num_patches_w))
    # print(f"{query_patches.shape=}")
    # print(f"{patches.shape=}")

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            target_patch = patches[:, :, i, j]
            sim = similarity(query_patches, target_patch)
            if sim is not None:
                similarity_matrix[:, i, j] = sim
            else:
                continue

    # 提取前k个最相似的图像块及其索引
    _, index = torch.topk(similarity_matrix.view(
        batch_size, -1), k, dim=-1, sorted=True)
    index_h = index // similarity_matrix.shape[2]  # [batch_size, k]
    index_w = index % similarity_matrix.shape[2]  # [batch_size, k]
    index_hw = index = torch.stack(
        [index_h, index_w], dim=-1)  # [batch_size, k, 2]
    top_k_patches = patches[torch.arange(similarity_matrix.shape[0])[
        :, None], :, index_h, index_w]

    return top_k_patches, index_hw


def convert_patches_to_bbox(patches, index_hw):
    """
    将图像块和对应的索引转换成边界框坐标

    参数:
    patches (torch.Tensor): 形状为 (batch_size, k, channels, patch_size, patch_size) 的图像块
    index_hw (torch.Tensor): 形状为 (batch_size, k, 2) 的图像块索引, 第二维代表(index_h, index_w)

    返回:
    bbox (torch.Tensor): 形状为 (batch_size, k, 4) 的边界框坐标, 最后一维表示(x, y, w, h)
    """
    batch_size, k, _, ph, pw = patches.shape

    # 从index_hw中获取index_h和index_w
    index_h, index_w = index_hw.unbind(-1)

    # 计算bbox的左上角坐标
    x = index_w
    y = index_h

    # 计算bbox的宽高
    w = torch.full((batch_size, k), pw)
    h = torch.full((batch_size, k), ph)

    # 将bbox左上角坐标转换成中心坐标
    x = x + w / 2
    y = y + h / 2

    # print(f"{x.shape=}")
    # print(f"{y.shape=}")
    # print(f"{w.shape=}")
    # print(f"{h.shape=}")

    # 将结果整理成(x, y, w, h)的形式
    bbox = torch.stack([x, y, w, h], dim=-1)

    return bbox


def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):
    """
    TODO: 需要针对CVOGL数据集进行修改
    """

    Q = len(query_features)
    steps = Q // step_size + 1
    similarity = []
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+1, dim=1)

    topk_references = []
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i, :]])
    topk_references = torch.stack(topk_references, dim=0)

    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()
    for i in range(len(topk_references)):
        nearest = topk_references[i][mask[i]][:neighbour_range]
        nearest_dict[query_labels[i].item()] = list(nearest)
    return nearest_dict
