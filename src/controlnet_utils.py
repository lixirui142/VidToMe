import torch.nn.functional as F
# from PyQt5.QtCore import QLibraryInfo
import cv2
import os
import torch
import torchvision.transforms as T
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
#     QLibraryInfo.PluginsPath
# )
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/lixirui/anaconda3/envs/dfwebui/lib/python3.9/site-packages/PyQt5/Qt5/plugins"

from controlnet_aux.processor import Processor
import transformers
import numpy as np
from diffusers.utils import load_image


processor_cache = dict()

def control_preprocess(images, control_type, **kwargs):
    if control_type == "tile":
        return tile_preprocess(images, **kwargs)
    elif control_type == "ip2p":
        return ip2p_prepreocess(images, **kwargs)
    elif control_type == "openpose":
        return openpose_prepreocess(images, **kwargs)
    elif "softedge" in control_type:
        return softedge_prepreocess(images, **kwargs)
    elif "depth" in control_type:
        return depth_prepreocess(images, **kwargs)
    elif control_type == "lineart_anime":
        return lineart_anime_prepreocess(images, **kwargs)
    elif control_type == "canny":
        return canny_preprocess(images, **kwargs)
    elif control_type == "none":
        return images


def tile_preprocess(image, resample_rate = 1.0, **kwargs):
    cond_image = F.interpolate(image, scale_factor=resample_rate, mode="bilinear")
    cond_image = F.interpolate(cond_image, scale_factor=1 / resample_rate)
    return cond_image
    
def ip2p_prepreocess(image, **kwargs):
    return image

def process(image, processor_id):
    process_ls = []
    H, W = image.shape[2:]
    if processor_id in processor_cache:
        processor = processor_cache[processor_id]
    else:
        processor = Processor(processor_id, {"output_type": "numpy"})
        processor_cache[processor_id] = processor
    for img in image:
        img = img.clone().cpu().permute(1,2,0) * 255
        processed_image = processor(img)
        processed_image = cv2.resize(processed_image, (W, H), interpolation=cv2.INTER_LINEAR)
        processed_image = torch.tensor(processed_image).to(image).permute(2,0,1) / 255
        process_ls.append(processed_image)
    processed_image = torch.stack(process_ls)
    return processed_image

def openpose_prepreocess(image, **kwargs):
    
    processor_id = 'openpose'

    return process(image, processor_id)

def softedge_prepreocess(image, proc = "pidsafe", **kwargs):

    # processor_id = 'softedge_hed'
    processor_id = f'softedge_{proc}'

    return process(image, processor_id)

def depth_prepreocess(image, **kwargs):
    
    image_ls = []
    for img in image:
        image_ls.append(T.ToPILImage()(img))
    depth_estimator = transformers.pipeline('depth-estimation')
    ret = depth_estimator(image_ls)
    depth_ls = []
    for r in ret:
        depth_ls.append(T.ToTensor()(r['depth']))

    depth = torch.cat(depth_ls)
    depth = torch.stack([depth, depth, depth], axis=1)
    return depth

def lineart_anime_prepreocess(image, proc = "anime",**kwargs):
    
    processor_id = f'lineart_{proc}'

    return process(image, processor_id)

def canny_preprocess(image, **kwargs):
    
    processor_id = f'canny'

    return process(image, processor_id)

def empty_cache():
    global processor_cache
    processor_cache = dict()
    torch.cuda.empty_cache()
