import time
import torch
from tqdm import tqdm
from .utils import AverageMeter, xyxy2xywh
from torch.cuda.amp import autocast
import torch.nn.functional as F

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, reference, ids, click_xy, gt_box in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                gt_box = torch.clamp(gt_box, min=0, max=train_config.img_size-1).to(train_config.device)
            
                # Forward pass
                features1, features2 = model(query, reference)
                # print(f"{features1.shape=}")
                # print(f"{features2.shape=}")

                scale1 = train_config.grd_size / features1.shape[2]
                scale2 = train_config.img_size / features2.shape[2]
                click_points = click_xy.clone().detach()
                click_points = click_points.to(train_config.device)
                # click_points = click_points // scale1
                click_points[:,0] = torch.div(click_points[:,0], scale1, rounding_mode='floor').long()
                click_points[:,1] = torch.div(click_points[:,1], scale1, rounding_mode='floor').long()

                # to xyxy2xywh
                gt_box = xyxy2xywh(gt_box)
                gt_box = gt_box / scale2
                gt_points = gt_box[:, 0:2].long()
                # print(f"{gt_points=}")

                patch_size = int(train_config.patch_size // scale2)
                # print(f"{patch_size=}")

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    # loss = loss_function(features1, features2, model.module.logit_scale.exp())
                    loss = loss_function(features1, features2, model.module.logit_scale.exp(), patch_size, click_points, gt_points)
                else:
                    # loss = loss_function(features1, features2, model.logit_scale.exp()) 
                    loss = loss_function(features1, features2, model.logit_scale.exp(), patch_size, click_points, gt_points)

                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)
            gt_box = torch.clamp(gt_box, min=0, max=train_config.img_size-1).to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)

            scale1 = train_config.grd_size / features1.shape[2]
            scale2 = train_config.img_size / features2.shape[2]
            click_points = torch.tensor(click_xy, device=train_config.device)
            click_points = click_points // scale1

            # to xyxy2xywh
            gt_box = xyxy2xywh(gt_box)
            gt_box = gt_box / scale2
            gt_points = gt_box[:, 0:2].long()
            # print(f"{gt_points=}")

            patch_size = int(train_config.patch_size // scale2)
            # print(f"{patch_size=}")

            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                # loss = loss_function(features1, features2, model.module.logit_scale.exp())
                loss = loss_function(features1, features2, model.module.logit_scale.exp(), patch_size, click_points, gt_points)
            else:
                # loss = loss_function(features1, features2, model.logit_scale.exp()) 
                loss = loss_function(features1, features2, model.logit_scale.exp(), patch_size, click_points, gt_points)
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    query_features_list = []
    ref_features_list = []
    click_xy_list = []
    gt_box_list = []
    # scale1 = 0
    # scale2 = 0
    
    ids_list = []
    with torch.no_grad():
        
        for query, reference, ids, click_xy, gt_box in bar:
        
            ids_list.append(ids)
            
            with autocast():
         
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                click_points = click_xy.clone().detach()
                click_points = click_points.to(train_config.device)
                gt_box = torch.clamp(gt_box, min=0, max=train_config.img_size-1).to(train_config.device)
            
                # Forward pass
                features1, features2 = model(query, reference)

                scale1 = train_config.grd_size / features1.shape[2]
                scale2 = train_config.img_size / features2.shape[2]
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    features1 = F.normalize(features1, dim=-1)
                    features2 = F.normalize(features2, dim=-1)
            
            # save features in fp32 for sim calculation
            query_features_list.append(features1.to(torch.float32))
            ref_features_list.append(features2.to(torch.float32))
            click_xy_list.append(click_points.to(torch.float32))
            gt_box_list.append(gt_box.to(torch.float32))
      
        # keep Features on GPU
        query_features = torch.cat(query_features_list, dim=0) 
        ref_features = torch.cat(ref_features_list, dim=0) 
        click_xy = torch.cat(click_xy_list, dim=0) 
        gt_box = torch.cat(gt_box_list, dim=0) 

        
    if train_config.verbose:
        bar.close()
        
    return query_features, ref_features, scale1, scale2, click_xy, gt_box 