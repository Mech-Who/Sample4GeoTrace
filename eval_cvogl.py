import os
import torch
from dataclasses import dataclass

from torch.utils.data import DataLoader
from sample4geo.dataset.cvogl import CVOGLDatasetEval
from sample4geo.transforms import get_transforms_val
from sample4geo.evaluate.cvogl import evaluate
from sample4geo.model import TimmModel


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    # img_size: int = 384
    img_size: int = 512
    grd_size: int = 256
    patch_size: int = 64

    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    
    # Dataset
    # data_folder = "/hong614/dataset/CVUSA"
    data_folder = "D:/Project/Reproduction/Datasets/CVOGL" 
    data_name = "CVOGL_DroneAerial"    
    
    # Checkpoint to start from
    # checkpoint_start = 'pretrained/cvusa/convnext_base.fb_in22k_ft_in1k_384/weights_e40_98.6830.pth' 
    checkpoint_start = 'pretrained/cvogl/convnext_base.fb_in22k_ft_in1k_384/135021/weights_e30_0.0130.pth'  
  
    # set num_workers to 0 if on Windows
    num_workers: int = 12 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    # model = TimmModel(config.model,
    #                   pretrained=True,
    #                   img_size=config.img_size)
    
    pretrained_cfg_overlay = {'file' : r"~/.cache/torch/hub/checkpoints/convnext_base.bin"}

    model = TimmModel(config.model,
                        # pretrained_cfg_overlay=pretrained_cfg_overlay,
                        pretrained_cfg_overlay=None,
                        pretrained=True,
                        img_size=config.img_size)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    
    image_size_sat = (img_size, img_size)
    if config.data_name == "CVOGL_DroneAerial":
        img_size_ground = (256, 256)
    elif config.data_name == "CVOGL_SVI":
        img_size_ground = (256, 512)
    
    # new_width = config.img_size * 1    
    # new_hight = round((img_size_ground[0] / img_size_ground[1]) * new_width)
    # img_size_ground = (new_hight, new_width)
     
    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#
        
    
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               mean=mean,
                                                               std=std,
                                                               )

    test_dataset = CVOGLDatasetEval(data_folder=config.data_folder,
                                    data_name=config.data_name,
                                    split="test",
                                    img_type="query",    
                                    transforms=(sat_transforms_val,ground_transforms_val)
                                    )
    
    test_dataloader = DataLoader(test_dataset,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                shuffle=False,
                                pin_memory=True)
    

    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#
    
    print("\n{}[{}]{}".format(30*"-", "CVOGL", 30*"-"))  

    r1_test = evaluate(config=config,
                    model=model,
                    dataloader=test_dataloader,
                )
    
    print(f"accu@0.5={r1_test[0]}, accu@0.25={r1_test[1]}")
