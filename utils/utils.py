import contextlib
import random
import numpy as np
import os
from glob import glob
from PIL import Image, ImageSequence

import torch
from torchvision.io import read_video, write_video
import torchvision.transforms as T

from diffusers import DDIMScheduler, StableDiffusionControlNetPipeline, StableDiffusionPipeline, StableDiffusionDepth2ImgPipeline, ControlNetModel
from .controlnet_utils import CONTROLNET_DICT, control_preprocess
from einops import rearrange

FRAME_EXT = [".jpg", ".png"]


def init_model(device="cuda", sd_version="1.5", model_key=None, control_type="none", weight_dtype="fp16"):

    use_depth = False
    if model_key is None:
        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif sd_version == 'depth':
            model_key = "stabilityai/stable-diffusion-2-depth"
            use_depth = True
        else:
            raise ValueError(
                f'Stable-diffusion version {sd_version} not supported.')

        print(f'[INFO] loading stable diffusion from: {model_key}')
    else:
        print(f'[INFO] loading custome model from: {model_key}')

    scheduler = DDIMScheduler.from_pretrained(
        model_key, subfolder="scheduler")

    if weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    if control_type not in ["none", "pnp"]:
        controlnet_key = CONTROLNET_DICT[control_type]
        print(f'[INFO] loading controlnet from: {controlnet_key}')
        controlnet = ControlNetModel.from_pretrained(
            controlnet_key, torch_dtype=weight_dtype)
        print(f'[INFO] loaded controlnet!')
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_key, controlnet=controlnet, torch_dtype=weight_dtype
        )
    elif use_depth:
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            model_key, torch_dtype=weight_dtype
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            # model_key, torch_dtype=weight_dtype
            model_key, torch_dtype=weight_dtype,
        )

    return pipe.to(device), scheduler, model_key


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = T.ToTensor()(image)
    return image.unsqueeze(0)


def process_frames(frames, h, w):

    fh, fw = frames.shape[-2:]
    h = int(np.floor(h / 64.0)) * 64
    w = int(np.floor(w / 64.0)) * 64

    nw = int(fw / fh * h)
    if nw >= w:
        size = (h, nw)
    else:
        size = (int(fh / fw * w), w)

    assert len(frames.shape) >= 3
    if len(frames.shape) == 3:
        frames = [frames]

    print(
        f"[INFO] frame size {(fh, fw)} resize to {size} and centercrop to {(h, w)}")

    frame_ls = []
    for frame in frames:
        resized_frame = T.Resize(size)(frame)
        cropped_frame = T.CenterCrop([h, w])(resized_frame)
        # croped_frame = T.FiveCrop([h, w])(resized_frame)[0]
        frame_ls.append(cropped_frame)
    return torch.stack(frame_ls)


def glob_frame_paths(video_path):
    frame_paths = []
    for ext in FRAME_EXT:
        frame_paths += glob(os.path.join(video_path, f"*{ext}"))
    frame_paths = sorted(frame_paths)
    return frame_paths


def load_video(video_path, h, w, frame_ids=None, device="cuda"):
    

    if ".mp4" in video_path:
        frames, _, _ = read_video(
            video_path, output_format="TCHW", pts_unit="sec")
        frames = frames / 255
    elif ".gif" in video_path:
        frames = Image.open(video_path)
        frame_ls = []
        for frame in ImageSequence.Iterator(frames):
            frame_ls += [T.ToTensor()(frame.convert("RGB"))]
        frames = torch.stack(frame_ls)
    else:
        frame_paths = glob_frame_paths(video_path)
        frame_ls = []
        for frame_path in frame_paths:
            frame = load_image(frame_path)
            frame_ls.append(frame)
        frames = torch.cat(frame_ls)
    if frame_ids is not None:
        frames = frames[frame_ids]

    print(f"[INFO] loaded video with {len(frames)} frames from: {video_path}")

    frames = process_frames(frames, h, w)
    return frames.to(device)


def save_video(frames: torch.Tensor, path, frame_ids=None, save_frame=False):
    os.makedirs(path, exist_ok=True)
    if frame_ids is None:
        frame_ids = [i for i in range(len(frames))]
    frames = frames[frame_ids]

    proc_frames = (rearrange(frames, "T C H W -> T H W C") * 255).to(torch.uint8).cpu()
    write_video(os.path.join(path, "output.mp4"), proc_frames, fps = 30, video_codec="h264")
    print(f"[INFO] save video to {os.path.join(path, 'output.mp4')}")

    if save_frame:
        save_frames(frames, os.path.join(path, "frames"), frame_ids = frame_ids)
    

def save_frames(frames: torch.Tensor, path, ext="png", frame_ids=None):
    os.makedirs(path, exist_ok=True)
    if frame_ids is None:
        frame_ids = [i for i in range(len(frames))]
    for i, frame in zip(frame_ids, frames):
        T.ToPILImage()(frame).save(
            os.path.join(path, '{:04}.{}'.format(i, ext)))


def load_latent(latent_path, t, frame_ids=None):
    latent_fname = f'noisy_latents_{t}.pt'

    lp = os.path.join(latent_path, latent_fname)
    assert os.path.exists(
        lp), f"Latent at timestep {t} not found in {latent_path}."

    latents = torch.load(lp)
    if frame_ids is not None:
        latents = latents[frame_ids]
    
    # print(f"[INFO] loaded initial latent from {lp}")

    return latents

@torch.no_grad()
def prepare_depth(pipe, frames, frame_ids, work_dir):
    
    depth_ls = []
    depth_dir = os.path.join(work_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    for frame, frame_id in zip(frames, frame_ids):
        depth_path = os.path.join(depth_dir, "{:04}.pt".format(frame_id))
        depth = load_depth(pipe, depth_path, frame)
        depth_ls += [depth]
    print(f"[INFO] loaded depth images from {depth_path}")
    return torch.cat(depth_ls)

# From pix2video: code/file_utils.py

def load_depth(model, depth_path, input_image, dtype=torch.float32):
    if os.path.exists(depth_path):
        depth_map = torch.load(depth_path)
    else:
        input_image = T.ToPILImage()(input_image.squeeze())
        depth_map = prepare_depth_map(
            model, input_image, dtype=dtype, device=model.device)
        torch.save(depth_map, depth_path)
        depth_image = (((depth_map + 1.0) / 2.0) * 255).to(torch.uint8)
        T.ToPILImage()(depth_image.squeeze()).convert(
            "L").save(depth_path.replace(".pt", ".png"))

    return depth_map

@torch.no_grad()
def prepare_depth_map(model, image, depth_map=None, batch_size=1, do_classifier_free_guidance=False, dtype=torch.float32, device="cuda"):
    if isinstance(image, Image.Image):
        image = [image]
    else:
        image = list(image)

    if isinstance(image[0], Image.Image):
        width, height = image[0].size
    elif isinstance(image[0], np.ndarray):
        width, height = image[0].shape[:-1]
    else:
        height, width = image[0].shape[-2:]

    if depth_map is None:
        pixel_values = model.feature_extractor(
            images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device)
        # The DPT-Hybrid model uses batch-norm layers which are not compatible with fp16.
        # So we use `torch.autocast` here for half precision inference.
        context_manger = torch.autocast(
            "cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
        with context_manger:
            ret = model.depth_estimator(pixel_values)
            depth_map = ret.predicted_depth
            # depth_image = ret.depth
    else:
        depth_map = depth_map.to(device=device, dtype=dtype)

    indices = depth_map != -1
    bg_indices = depth_map == -1
    min_d = depth_map[indices].min()

    if bg_indices.sum() > 0:
        depth_map[bg_indices] = min_d - 10
        # min_d = min_d - 10

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(height // model.vae_scale_factor,
              width // model.vae_scale_factor),
        mode="bicubic",
        align_corners=False,
    )

    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
    depth_map = depth_map.to(dtype)

    # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
    if depth_map.shape[0] < batch_size:
        repeat_by = batch_size // depth_map.shape[0]
        depth_map = depth_map.repeat(repeat_by, 1, 1, 1)

    depth_map = torch.cat(
        [depth_map] * 2) if do_classifier_free_guidance else depth_map
    return depth_map


def get_latents_dir(latents_path, model_key):
    model_key = model_key.split("/")[-1]
    return os.path.join(latents_path, model_key)


def get_controlnet_kwargs(controlnet, x, cond, t, controlnet_cond, controlnet_scale=1.0):
    down_block_res_samples, mid_block_res_sample = controlnet(
        x,
        t,
        encoder_hidden_states=cond,
        controlnet_cond=controlnet_cond,
        return_dict=False,
    )
    down_block_res_samples = [
        down_block_res_sample * controlnet_scale
        for down_block_res_sample in down_block_res_samples
    ]
    mid_block_res_sample *= controlnet_scale
    controlnet_kwargs = {"down_block_additional_residuals": down_block_res_samples,
                         "mid_block_additional_residual": mid_block_res_sample}
    return controlnet_kwargs


def get_frame_ids(frame_range, frame_ids=None):
    if frame_ids is None:
        frame_ids = list(range(*frame_range))
    frame_ids = sorted(frame_ids)

    if len(frame_ids) > 4:
        frame_ids_str = "{} {} ... {} {}".format(
            *frame_ids[:2], *frame_ids[-2:])
    else:
        frame_ids_str = " ".join(["{}"] * len(frame_ids)).format(*frame_ids)
    print("[INFO] frame indexes: ", frame_ids_str)
    return frame_ids


def prepare_control(control, frames, frame_ids, save_path):
    if control not in CONTROLNET_DICT.keys():
        print(f"[WARNING] unknown controlnet type {control}")
        return None

    control_subdir = f'{save_path}/{control}_image'

    preprocess_flag = True
    if os.path.exists(control_subdir):
        print(f"[INFO] load control image from {control_subdir}.")
        control_image_ls = []
        for frame_id in frame_ids:
            image_path = os.path.join(
                control_subdir, "{:04}.png".format(frame_id))
            if not os.path.exists(image_path):
                break
            control_image_ls += [load_image(image_path)]
        else:
            preprocess_flag = False
            control_images = torch.cat(control_image_ls)

    if preprocess_flag:
        print("[INFO] preprocessing control images...")
        control_images = control_preprocess(frames, control)
        print(f"[INFO] save control images to {control_subdir}.")
        os.makedirs(control_subdir, exist_ok=True)
        for image, frame_id in zip(control_images, frame_ids):
            image_path = os.path.join(
                control_subdir, "{:04}.png".format(frame_id))
            T.ToPILImage()(image).save(image_path)

    return control_images
