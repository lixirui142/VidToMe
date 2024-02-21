import glob
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm

from diffusers import DDIMScheduler, StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionDepth2ImgPipeline
from torch.optim import Adam, AdamW
from controlnet_utils import control_preprocess, empty_cache
# from transformers import logging
from smooth import smooth_seq
from pnp_utils import *
from scripts.core.flow_utils import RAFT_estimate_flow, compute_occ_mask_tensor, warp_frame
from scripts.utils import load_depth, switch_format
import tomesd
from diffusers.utils import randn_tensor
from scripts.video_smooth import blur_video
# suppress partial model loading warning
# logging.set_verbosity_error()
from torchvision.utils import flow_to_image
from transformers import AutoProcessor, CLIPModel
class PNP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        sd_version = config["sd_version"]


        self.use_depth = False
        if "model_key" in config:
            model_key = config["model_key"]
        else:
            if sd_version == '2.1':
                model_key = "stabilityai/stable-diffusion-2-1-base"
            elif sd_version == '2.0':
                model_key = "stabilityai/stable-diffusion-2-base"
            elif sd_version == '1.5':
                model_key = "runwayml/stable-diffusion-v1-5"
            elif sd_version == 'depth':
                model_key = "stabilityai/stable-diffusion-2-depth"
                self.use_depth = True
            else:
                raise ValueError(
                    f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        control_paths = {"tile": "lllyasviel/control_v11f1e_sd15_tile", "ip2p": "lllyasviel/control_v11e_sd15_ip2p",
                         "openpose": "lllyasviel/control_v11p_sd15_openpose", "softedge": "lllyasviel/control_v11p_sd15_softedge",
                         "depth": "lllyasviel/control_v11f1p_sd15_depth", "lineart_anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
                         "canny": "lllyasviel/control_v11p_sd15_canny", "depth_dinv": "lllyasviel/control_v11f1p_sd15_depth",
                         "softedge_sinv": "lllyasviel/control_v11p_sd15_softedge"}

        #model_dir = "/home/lixirui/Repos/stable-diffusion-webui/extensions/sd-webui-controlnet/models"
        #tile_model = os.path.join(model_dir, "control_v11f1e_sd15_tile.pth")
        control_type = config["control"]
        self.control = control_type
        self.model_key = model_key

        if control_type != "none" and "pnp" not in control_type:
            control_model = control_paths[control_type]

            print('Loading ControlNet model')
            print(control_model)
            controlnet = ControlNetModel.from_pretrained(
                control_model, torch_dtype=torch.float16, local_files_only=True)

        control_type2 = config["multi_control"]
        self.control2 = control_type2

        print('Loading SD model')
        print(model_key)
        if control_type != "none" and "pnp" not in control_type:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_key, controlnet=controlnet, torch_dtype=torch.float16, cache_dir="/home/lixirui/.cache/huggingface/diffusers", local_files_only=True
            ).to("cuda")
        elif self.use_depth:
            pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
                model_key, torch_dtype=torch.float16, cache_dir="/home/lixirui/.cache/huggingface/diffusers",
                local_files_only = True
            ).to("cuda")
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_key, torch_dtype=torch.float16, cache_dir="/home/lixirui/.cache/huggingface/diffusers", local_files_only=True
            ).to("cuda")

        if control_type2 != "none" and control_type2 != "pnp":
            control_model2 = control_paths[control_type2]

            print('Loading Second ControlNet model')
            print(control_model2)
            controlnet2 = ControlNetModel.from_pretrained(
                control_model2, torch_dtype=torch.float16)
            self.controlnet2 = controlnet2.to("cuda")
            pipe.controlnet2 = self.controlnet2

        self.controls = [self.control, self.control2]
        self.pnp = "pnp" in self.control or "pnp" in self.control2
        # pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16,
            # cache_dir="/home/lixirui/.cache/huggingface/diffusers", local_files_only=True).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.merge_crossattn = False
        self.merge_mlp = False
        self.pipe = pipe
        self.batch_size = 3 if self.pnp else 2
        self.save_pca = config["save_pca"] if "save_pca" in config else False
        self.save_tome = config["save_tome"] if "save_tome" in config else False
        self.include_control = False
        self.coord_mask = False
        self.merge_global = True
        
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        if self.control != "none" and "pnp" not in self.control:
            self.controlnet = pipe.controlnet

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", local_files_only = True)
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)
        print('SD model loaded')
        if "crop" in config:
            self.crop = config["crop"]
        else:
            self.crop = False
        self.start_step = 0
        self.img_indexes = torch.tensor(self.config["img_indexes"], dtype=torch.long)
        # load image
        self.eps = self.get_data()

        self.ratio = self.config["ratio"] if "ratio" in self.config else .9
        self.global_rand = self.config["global_rand"] if "global_rand" in self.config else .5
        self.global_merge_ratio = self.config["global_merge_ratio"] if "global_merge_ratio" in self.config else .8


        f = len(self.eps)
        if f > 1:
            self.activate_tome()

        

            #self.eps = (self.eps - self.eps.mean()) / self.eps.std()
            # self.eps[:] = self.eps[0:1]
        # prev_edit_image = self.load_image(config["prev_edit_image_path"])

        # self.save_prev_warp(prev_edit_image)
        # exit(0)

        self.text_embeds = self.get_text_embeds(
            config["prompt"], config["negative_prompt"])
        # self.text_embeds = self.get_text_embeds("","")
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]
        # self.pnp_guidance_embeds = self.get_text_embeds("a dog is walking on the ground", "a dog is walking on the ground").chunk(2)[0]
        # a dog is walking on the ground
        self.save_path = os.path.join(
            config["latents_path"] + "G", config["suffix"])

        os.makedirs(self.save_path, exist_ok=True)

        # self.optimizer = Adam(self.vae.parameters(), lr=0.0)
        self.save = config["save"]
        if self.save:
            print(f"Intermediate Latent Saved to {self.save_path}")

        # self.controlnet_conditioning_scale = 0.6
        self.controlnet_conditioning_scale = config["cond_scale"]
        self.controlnet_conditioning_scale2 = config["cond_scale2"] if "cond_scale2" in config else None
        self.control_trange = [0, 1000]
        self.multi_frames = config["multi_frames"]

        if self.pnp:
            self.max_num_kf = 4
            self.kf_num = max( len(self.eps) // 8, 8)
            self.kf_interval = len(self.eps) // self.kf_num
            self.max_num_int = 4
        else:
            self.max_num_kf = 4
            self.kf_num = max( len(self.eps) // 8, 8)
            self.kf_interval = len(self.eps) // self.kf_num
            self.max_num_int = 4

        self.max_num_vae = 4

        if self.merge_global:
            self.kf_num = 0

        self.dynamic_ctr = False
        if "save_intermediate" in config:
            self.save_intermediate = config["save_intermediate"]
        else:
            self.save_intermediate = False
        if "use_warp_smooth" in config:
            self.use_warp_smooth = config["use_warp_smooth"]
        else:
            self.use_warp_smooth = False
        if "use_blur_smooth" in config:
            self.use_blur_smooth = config["use_blur_smooth"]
        else:
            self.use_blur_smooth = False

        self.max_warp_smooth = 1000
        self.min_warp_smooth = 200
        self.max_blur_smooth = 1000
        self.min_blur_smooth = 500

        self.chunk_ord = config["chunk_ord"] if "chunk_ord" in self.config else "mix-4" # seq, perm, mix
        if "mix" in self.chunk_ord:
            self.chunk_ord = "mix"
            if "-" in self.chunk_ord:
                self.perm_div = float(self.chunk_ord.split("-")[-1])


        rand_initial = True
        edit_initial = False
        directional_rand_initial = False
        gradually_change_initial = False
        if rand_initial:
            # ratio = 1.0
            # c, h, w = self.encode_imgs(self.image[0:1]).shape[1:]
            # self.eps = torch.randn([f, c, h, w]).to(self.eps)
            fix_initial = False
            if fix_initial:
                self.eps = torch.randn_like(self.eps)[[0]].expand(f, -1, -1, -1)
            else:
                self.eps = torch.randn_like(self.eps)
        elif edit_initial:
            fix_noise = False
            source_latents = self.encode_imgs_chunk(self.source_image)
            self.eps = self.forward_sample(source_latents, self.scheduler.timesteps[self.start_step], fix_noise=fix_noise, eps = self.eps)
        elif directional_rand_initial:
            move_strength = 0.5
            neps = torch.randn_like(self.eps[[0]]).expand(f, -1, -1, -1)
            direction = (self.eps - self.eps[[0]].expand(f, -1, -1, -1))
            direction = direction / direction.norm(dim=1,keepdim=True).clamp(min=1e-10)
            neps = neps + direction * move_strength
            self.eps = neps
        elif gradually_change_initial:
            neps = [torch.randn_like(self.eps[[0]])]
            t = self.scheduler.timesteps[3]
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_T = self.scheduler.alphas_cumprod[981]
            coef0 = (alpha_prod_T / alpha_prod_t) ** 0.5
            coef1 = (1 - alpha_prod_T / alpha_prod_t) ** 0.5

            for i in range(len(self.eps) - 1):
                prev_eps = neps[-1]
                cur_eps = prev_eps * coef0 + torch.randn_like(self.eps[[0]]) * coef1
                neps += [cur_eps]
            neps = torch.cat(neps)
            self.eps = neps



        # preatrained_model_path = "openai/clip-vit-large-patch14"
        # clip_model = CLIPModel.from_pretrained(preatrained_model_path, local_files_only=True).to(self.device)
        # clip_processor = AutoProcessor.from_pretrained(preatrained_model_path, local_files_only=True)
        # image_tensors = T.FiveCrop(256)(self.image[0])
        # image_inputs = clip_processor(images=image_tensors, return_tensors="pt").to(self.device)
        # text_inputs = clip_processor(text=[config["prompt"]], return_tensors="pt").to(self.device)

        # # textf = clip_model.get_text_features(**text_inputs)
        # imgf = clip_model.get_image_features(**image_inputs)



        self.control_drop_ratio = None
        self.vae = self.vae.to("cpu")
        
        # self.pipe.load_lora_weights("artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5", weight_name="PixelArtRedmond15V-PixelArt-PIXARFK.safetensors", adapter_name="pixel")
        if "lora" in config:
            self.pipe.load_lora_weights(config["lora"], weight_name=config["lora_weight_name"], adapter_name=config["lora_adapter"], adapter_weights=config["lora_weight"])

            # self.pipe.set_adapters([config["lora_adapter"]], adapter_weights=[config["lora_weight"]])


    def activate_tome(self):
        # 6 / 7
        tomesd.apply_patch(self.pipe, ratio=self.ratio, merge_attn=True, merge_all=True, merge_adj=True, hier_merge=True, merge_to=1, max_downsample=2, max_downsample_join=2,
                           merge_mlp=self.merge_mlp, merge_crossattn=self.merge_crossattn, batch_size=self.batch_size, include_control=self.include_control, 
                           store_pca=self.save_pca, store_tome = self.save_tome,coord_mask=self.coord_mask, init_inframe_tome = False, merge_global= self.merge_global, global_merge_ratio=self.global_merge_ratio,
                           global_rand = self.global_rand)
        # tomesd.apply_patch(self.pipe, only_join = True, batch_size=self.batch_size, merge_to=1, max_downsample=2, max_downsample_join=2)
        # tomesd.apply_patch(self.pipe, ratio=2.7 / 4.0, join_after_merge=True, max_downsample=2, max_downsample_join=2, batch_size=self.batch_size)
        # tomesd.apply_patch(self.pipe, ratio=2.7 / 4.0, merge_all=True, max_downsample=2, max_downsample_join=2, batch_size=self.batch_size)
        tomesd.update_patch(self.pipe, adhere_src=self.pnp)

    def remove_tome(self):
        tomesd.remove_patch(self.pipe)

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device))[0]

        # text_embeddings[:, [8,9]] *= 1.5

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat(
            [uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    def decode_latent_with_grad(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def encode_imgs_chunk(self, imgs):
        latents = []
        fnum = len(imgs)
        slices = [k for k in range(0, fnum, self.max_num_vae)]
        for s in slices:
            latents += [self.encode_imgs(imgs[s:s+self.max_num_vae])]
        latents = torch.cat(latents)
        return latents

    def load_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        # image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.Resize(512)(image)
        H, W = image.height, image.width
        H = int(np.floor(H / 64.0)) * 64
        W = int(np.floor(W / 64.0)) * 64
        # Tmp
        H, W = min(H, W), min(H, W)
        if self.crop:
            image = T.FiveCrop(min(H, W))(image)[0]
        else:
            image = T.CenterCrop([H, W])(image)
        image = T.ToTensor()(image).to(self.device)
        return image.unsqueeze(0)

    def load_latent(self, t = None, step = None):
        assert t is not None or step is not None
        if t is None:
            t = self.scheduler.timesteps[step]
        latent_fname = f'noisy_latents_{t}.pt'
        
        lp = os.path.join(self.config["latents_path"], latent_fname)
        if os.path.exists(lp):
            latents = torch.load(lp)
            latents = latents[self.img_indexes]
        else:
            latents_ls = []
            for img_path in self.config["image_path_list"]:
                latents_path_o = os.path.join(self.config["latents_path"], os.path.splitext(
                    os.path.basename(img_path))[0], latent_fname)
                noisy_latent_o = torch.load(latents_path_o).to(self.device)
                latents_ls.append(noisy_latent_o)
            latents = torch.cat(latents_ls)
        return latents



    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self):
        # load image
        img_ls = []
        depth_ls = []
        
        print(
            f'Load Initial Noise from {self.config["latents_path"]}, {self.scheduler.timesteps[self.start_step]}.')
        for img_path in self.config["image_path_list"]:


            
            img = self.load_image(img_path)
            if self.use_depth:
                ext = img_path.split(".")[-1]
                depth_path = os.path.join(os.path.dirname(img_path), "depth", os.path.basename(img_path).replace(f".{ext}", ".pt"))
                os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                # np_image = np.array(T.ToPILImage()(image))
                # with torch.autocast(device_type='cuda', dtype=torch.float32):
                depth = load_depth(self.pipe, depth_path, img)
                depth_ls += [depth]
            img_ls.append(img)
        self.source_image = torch.cat(img_ls).to("cpu")
        if self.use_depth:

            # self.depths = control_preprocess(
            #         self.source_image, "depth")[:,[0],...]
            # depth_map = torch.nn.functional.interpolate(
            #     self.depths,
            #     size=(512 // 8, 512 // 8),
            #     mode="bicubic",
            #     align_corners=False,
            # )

            # depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            # depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            # depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
            # # depth_map = depth_map.to(dtype)
            # self.depths = depth_map

            self.depths = torch.cat(depth_ls)
            # self.pipe.depth_estimator = self.pipe.depth_estimator.to("cpu")
            # self.pipe.feature_extractor = self.pipe.feature_extractor.to("cpu")


        noisy_latent = self.load_latent(step = self.start_step)
        #H, W = noisy_latent.shape[2:]
        #H = int(np.floor(H / 64.0)) * 64
        #W = int(np.floor(W / 64.0)) * 64
        #noisy_latent = T.FiveCrop(min(H,W))(noisy_latent)[0]
        #print(f"Initial Noise Left Crop to {min(H,W)}")
        self.image = self.load_cond(self.control)
        if hasattr(self, "control2") and self.control2 != "none":
            self.image2 = self.load_cond(self.control2)
        
        # empty_cache()
        return noisy_latent

    def load_cond(self, control):
        if control != "none" and "pnp" not in control:
            image = self.source_image

            subdir = f"{control}"
            input_dict = {}
            if "cond_processor" in self.config:
                proc = self.config["cond_processor"]
                subdir += f"_{proc}"
                input_dict["proc"] = proc

            control_subdir = f'{self.config["output_path"]}/{subdir}_image'

            if os.path.exists(control_subdir):
                print(f"Load control image from {control_subdir}")
                image_list = glob.glob(os.path.join(control_subdir, "*"))
                image_list = sorted(image_list)
                cond_image = []
                for imgpath in image_list:
                    cond_image.append(self.load_image(imgpath))
                cond_image = torch.cat(cond_image)
            else:

                print("Preprocessing control images")
                cond_image = control_preprocess(
                    image, control, **input_dict)
                print(f"Save control image from {control_subdir}")
                os.makedirs(control_subdir)
                for pth, ci in zip(self.config["image_path_list"], cond_image):
                    save_pth = f'{control_subdir}/{os.path.splitext(os.path.basename(pth))[0]}_control.png'
                    T.ToPILImage()(ci).save(save_pth)
            empty_cache()

            # height, width = self.pipe._default_height_width(None, None, cond_image)
            batch_size = len(image)
            num_images_per_prompt = 1
            device = image.device
            cond_image = self.pipe.prepare_image(
                cond_image,
                None,
                None,
                batch_size * num_images_per_prompt,
                num_images_per_prompt,
                device,
                self.controlnet.dtype,
            )
        else:
            cond_image = self.source_image
        return cond_image

    @torch.no_grad()
    def split_denoise_step(self, x, t):
        # register the time step and features in pnp injection modules
        source_latents_ls = []
        for img_path in self.config["image_path_list"]:
            source_latents = load_source_latents_t(t, os.path.join(
                self.config["latents_path"], os.path.splitext(os.path.basename(img_path))[0]))
            source_latents_ls.append(source_latents)
        source_latents = torch.cat(source_latents_ls)

        f = len(self.config["image_path_list"])
        # scale_factor = x.shape[2] / self.prev_flow.shape[0]
        # prev_flow_downsampled = F.interpolate(switch_format(self.prev_flow), scale_factor=scale_factor, mode='bilinear', align_corners=True) * scale_factor
        # warped_prev_latents = switch_format(warp_frame(switch_format(prev_flow_downsampled), switch_format(prev_latents)), device=self.device)

        # latent_model_input = torch.cat([source_latents] + ([x] * 2) + [warped_prev_latents])
        # latent_model_input = torch.cat([source_latents] + ([x] * 2) + [prev_latents])
        latent_model_input = torch.cat(
            [source_latents, x])

        register_time(self, t.item())

        # compute text embeddings
        # if not self.merge_crossattn:
        pnp_embeds = torch.cat([self.pnp_guidance_embeds] * f)
        text_embeds = self.text_embeds.repeat_interleave(f, dim=0)
        neg_embeds, pos_embeds = text_embeds.chunk(2)

        neg_text_embed_input = torch.cat(
            [pnp_embeds, neg_embeds], dim=0)
        pos_text_embed_input = torch.cat(
            [pnp_embeds, pos_embeds], dim=0)
        # else:
        # text_embed_input = torch.cat(
        # [self.pnp_guidance_embeds, self.text_embeds], dim=0)
        # text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds, self.prev_guidance_embeds.chunk(2)[0]], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t,
                               encoder_hidden_states=neg_text_embed_input)['sample']

        _, noise_pred_uncond = noise_pred.chunk(2)

        noise_pred = self.unet(latent_model_input, t,
                               encoder_hidden_states=pos_text_embed_input)['sample']

        _, noise_pred_cond = noise_pred.chunk(2)

        # perform guidance
        noise_pred = noise_pred_uncond + \
            self.config["guidance_scale"] * \
            (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    @torch.no_grad()
    def denoise_step(self, x, t, image=None, image_idx=None):
        # register the time step and features in pnp injection modules

        if "cond_path" in self.config:
            source_latents = load_source_latents_t(t, self.config["cond_path"])
            cond_pos = self.config["cond_pos"].to(source_latents.device)
            prev_cond_pos = self.config["prev_cond_pos"].to(
                source_latents.device)
            assert len(cond_pos) == len(prev_cond_pos)
            x[cond_pos] = source_latents[prev_cond_pos]

        f = len(x)
        
        # scale_factor = x.shape[2] / self.prev_flow.shape[0]
        # prev_flow_downsampled = F.interpolate(switch_format(self.prev_flow), scale_factor=scale_factor, mode='bilinear', align_corners=True) * scale_factor
        # warped_prev_latents = switch_format(warp_frame(switch_format(prev_flow_downsampled), switch_format(prev_latents)), device=self.device)

        # latent_model_input = torch.cat([source_latents] + ([x] * 2) + [warped_prev_latents])
        # latent_model_input = torch.cat([source_latents] + ([x] * 2) + [prev_latents])
        latent_model_input = torch.cat([x, x])

        # compute text embeddings
        # if not self.merge_crossattn:
        text_embed_input = torch.cat(
            [self.text_embeds.repeat_interleave(f, dim=0)], dim=0)
        # text_embed_input = self.text_embeds

        # else:
        # text_embed_input = torch.cat(
        # [self.pnp_guidance_embeds, self.text_embeds], dim=0)
        # text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds, self.prev_guidance_embeds.chunk(2)[0]], dim=0)

        if t > self.control_tcut and self.control != "none":
            if image is None:
                image = self.image
            image = torch.cat([image, image])
            controlnet_conditioning_scale = self.controlnet_conditioning_scale
            if self.dynamic_ctr:
                controlnet_conditioning_scale = controlnet_conditioning_scale * t / self.scheduler.timesteps.max()
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embed_input,
                controlnet_cond=image,
                return_dict=False,
            )

            down_block_res_samples = [
                down_block_res_sample * controlnet_conditioning_scale
                for down_block_res_sample in down_block_res_samples
            ]
            mid_block_res_sample *= controlnet_conditioning_scale

            if hasattr(self, "control2") and self.control2 != "none":
                image2 = self.image2
                image2 = torch.cat([image2, image2])
                controlnet_conditioning_scale2 = self.controlnet_conditioning_scale2

                if self.dynamic_ctr:
                    controlnet_conditioning_scale = controlnet_conditioning_scale * t / self.scheduler.timesteps.max()
                down_block_res_samples2, mid_block_res_sample2 = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embed_input,
                    controlnet_cond=image2,
                    return_dict=False,
                )

                down_block_res_samples2 = [
                    down_block_res_sample2 * controlnet_conditioning_scale2
                    for down_block_res_sample2 in down_block_res_samples2
                ]
                mid_block_res_sample2 *= controlnet_conditioning_scale2



            # apply the denoising network
            noise_pred = self.unet(latent_model_input, t,
                                   encoder_hidden_states=text_embed_input,
                                   down_block_additional_residuals=down_block_res_samples,
                                   mid_block_additional_residual=mid_block_res_sample)['sample']
        else:

            noise_pred = self.unet(latent_model_input, t,
                                   encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + \
            self.config["guidance_scale"] * \
            (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']

        if self.save_intermediate:
            self.save_intermediate_x0(x, noise_pred, t, image_idx)

        return denoised_latent

    @torch.no_grad()
    def pred_noise(self, x, t, image_idx=None):
        # register the time step and features in pnp injection modules

        if "cond_path" in self.config:
            source_latents = load_source_latents_t(t, self.config["cond_path"])
            cond_pos = self.config["cond_pos"].to(source_latents.device)
            prev_cond_pos = self.config["prev_cond_pos"].to(
                source_latents.device)
            assert len(cond_pos) == len(prev_cond_pos)
            x[cond_pos] = source_latents[prev_cond_pos]
        f = len(x)
        
        if self.pnp:
            register_time(self, t.item())
            if image_idx is None:
                source_latents = self.cur_eps
            else:
                source_latents = self.cur_eps[image_idx]
            latent_model_input = torch.cat([source_latents.to(x), x, x])
            text_embed_input = torch.cat(
                [self.pnp_guidance_embeds, self.text_embeds], dim=0)
            text_embed_input = text_embed_input.repeat_interleave(f, dim=0)
        else:
            latent_model_input = torch.cat([x, x])

            # compute text embeddings
            # if not self.merge_crossattn:
            text_embed_input = torch.cat(
                [self.text_embeds.repeat_interleave(f, dim=0)], dim=0)
            
        if self.use_depth:
            depth = self.depths
            if image_idx is not None:
                depth = depth[image_idx]
            depth = depth.repeat(self.batch_size, 1, 1, 1)
            latent_model_input = torch.cat([latent_model_input, depth.to(latent_model_input)], dim = 1)

        # scale_factor = x.shape[2] / self.prev_flow.shape[0]
        # prev_flow_downsampled = F.interpolate(switch_format(self.prev_flow), scale_factor=scale_factor, mode='bilinear', align_corners=True) * scale_factor
        # warped_prev_latents = switch_format(warp_frame(switch_format(prev_flow_downsampled), switch_format(prev_latents)), device=self.device)

        # latent_model_input = torch.cat([source_latents] + ([x] * 2) + [warped_prev_latents])
        # latent_model_input = torch.cat([source_latents] + ([x] * 2) + [prev_latents])

        # text_embed_input = self.text_embeds

        # else:
        # text_embed_input = torch.cat(
        # [self.pnp_guidance_embeds, self.text_embeds], dim=0)
        # text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds, self.prev_guidance_embeds.chunk(2)[0]], dim=0)

        if t > self.control_trange[0] and t < self.control_trange[1] and self.control != "none" and "pnp" not in self.control:
            if image_idx is None:
                image = self.image
            else:
                image = self.image[image_idx]
            image = torch.cat([image, image])
            image = image.to(latent_model_input)
            controlnet_conditioning_scale = self.controlnet_conditioning_scale
            if self.dynamic_ctr:
                controlnet_conditioning_scale = self.controlnet_conditioning_scale * t / self.scheduler.timesteps.max()
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embed_input,
                controlnet_cond=image,
                return_dict=False,
            )

            down_block_res_samples = [
                down_block_res_sample * controlnet_conditioning_scale
                for down_block_res_sample in down_block_res_samples
            ]
            mid_block_res_sample *= controlnet_conditioning_scale


            
            if hasattr(self, "control2") and self.control2 != "none" and self.control2 != "pnp":
                image2 = self.image2
                image2 = torch.cat([image2, image2])
                controlnet_conditioning_scale2 = self.controlnet_conditioning_scale2

                if self.dynamic_ctr:
                    controlnet_conditioning_scale = controlnet_conditioning_scale * t / self.scheduler.timesteps.max()
                torch.cuda.empty_cache()
                down_block_res_samples2, mid_block_res_sample2 = self.controlnet2(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embed_input,
                    controlnet_cond=image2,
                    return_dict=False,
                )

                down_block_res_samples =[
                    down_block_res_sample2 * controlnet_conditioning_scale2 + down_block_res_sample
                    for down_block_res_sample, down_block_res_sample2 in zip(down_block_res_samples, down_block_res_samples2)
                ]
                mid_block_res_sample = mid_block_res_sample2 * controlnet_conditioning_scale2 + mid_block_res_sample

            if self.control_drop_ratio is not None:
                dropout = (torch.rand(f) < self.control_drop_ratio).repeat(self.batch_size)
                for j, dbrs in enumerate(down_block_res_samples):
                    down_block_res_samples[j][dropout] = 0

                mid_block_res_sample[dropout] = 0

            # apply the denoising network
            noise_pred = self.unet(latent_model_input, t,
                                   encoder_hidden_states=text_embed_input,
                                   down_block_additional_residuals=down_block_res_samples,
                                   mid_block_additional_residual=mid_block_res_sample)['sample']


        else:

            noise_pred = self.unet(latent_model_input, t,
                                   encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        if self.pnp:
            _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        else:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + \
            self.config["guidance_scale"] * \
            (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model

        return noise_pred

    def pred_original(self, x, t, eps):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        mu = alpha_prod_t ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        pred_x0 = (x - sigma * eps) / mu
        return pred_x0

    def forward_sample(self, x0, t, eps = None, fix_noise = False):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        mu = alpha_prod_t ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        if eps is None:
            eps = torch.randn_like(x0)
            if fix_noise:
                eps = eps[[0]].expand(len(x0), -1, -1, -1)
        # pred_x0 = (x - sigma * eps) / mu
        sample_xt = mu * x0 + sigma * eps
        return sample_xt

    # def denoise_step_cguidance(self, x, t):
    #     x = torch.nn.Parameter(x, requires_grad=True)
    #     # register the time step and features in pnp injection modules
    #     source_latents = load_source_latents_t(t, os.path.join(
    #         self.config["latents_path"], os.path.splitext(os.path.basename(self.config["image_path"]))[0]))
    #     prev_latents = load_source_latents_t(t, os.path.join(
    #         self.config["prev_latents_path"], os.path.splitext(os.path.basename(self.config["prev_image_path"]))[0]))

    #     # scale_factor = x.shape[2] / self.prev_flow.shape[0]
    #     # prev_flow_downsampled = F.interpolate(switch_format(self.prev_flow), scale_factor=scale_factor, mode='bilinear', align_corners=True) * scale_factor
    #     # warped_prev_latents = switch_format(warp_frame(switch_format(prev_flow_downsampled), switch_format(prev_latents)), device=self.device)

    #     # latent_model_input = torch.cat([source_latents] + ([x] * 2) + [warped_prev_latents])
    #     latent_model_input = torch.cat(
    #         [source_latents] + ([x] * 2) + ([prev_latents] * 2))

    #     register_time(self, t.item())
    #     register_flow(self, self.prev_flow, self.occlusion_mask)

    #     # compute text embeddings
    #     text_embed_input = torch.cat(
    #         [self.pnp_guidance_embeds, self.text_embeds, self.prev_guidance_embeds], dim=0)

    #     # apply the denoising network
    #     with torch.no_grad():
    #         noise_pred = self.unet(
    #             latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

    #     # perform guidance
    #     _, noise_pred_uncond, noise_pred_cond, _, _ = noise_pred.chunk(5)
    #     noise_pred = noise_pred_uncond + \
    #         self.config["guidance_scale"] * \
    #         (noise_pred_cond - noise_pred_uncond)

    #     pred_x0 = self.pred_original(x, t, noise_pred)
    #     pred_x0 = self.decode_latent_with_grad(pred_x0)
    #     loss = torch.abs(pred_x0 - self.pred_image_edited_warped).mean()
    #     loss.backward()

    #     # compute the denoising step with the reference model
    #     denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']

    #     denoised_latent = denoised_latent - x.grad * self.config["cg"]
    #     self.optimizer.zero_grad()
    #     return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else [
        ]
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else [
        ]
        register_attention_control_efficient(
            self, self.qk_injection_timesteps, num_inputs=self.batch_size)
        register_conv_control_efficient(
            self, self.conv_injection_timesteps, num_inputs=self.batch_size)
        # register_prev_control_efficient_att(self, self.conv_injection_timesteps)
        # register_prev_control_efficient_attmap_warp(self, self.qk_injection_timesteps, num_inputs=5)
        # register_prev_control_efficient(self, self.conv_injection_timesteps)
        # register_prev_control_efficient_featwarp(self, self.conv_injection_timesteps, num_inputs=5)

    def run_pnp(self):
        if self.pnp:

            pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
            pnp_attn_t = int(self.config["n_timesteps"]
                           * self.config["pnp_attn_t"])
            self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)

        source_pth = self.config["latents_path"]
        print(f"Source latents Load from {source_pth}")
        # self.pipe = self.pipe.to("cpu")
        self.unet = self.unet.to("cuda")
        self.eps = self.eps.to(torch.float16)
        if self.config["multi_frames"]:
            
            edited_img = self.multi_frame_sample_loop(self.eps)
        else:
            edited_img = self.sample_loop(self.eps)

    def save_noise(self):
        cur_latents = self.load_latent(t = 501)
        save_path = self.config["latents_path"] + "noise_image"
        self.vae = self.vae.to(cur_latents)
        self.decode_and_save(cur_latents, path = save_path)

    def get_image_names(self):
        image_names = []
        for img_path in self.config["image_path_list"]:
            image_names += [os.path.splitext(os.path.basename(img_path))[0]]
        return image_names

    def decode_and_save(self, x=None, path=None, image_idx=None, decoded_latent=None):

        if path is None:
            path = self.config["image_save_path"]

        assert x is not None or decoded_latent is not None, "Please make sure to pass one of the latents to decode or decoded latent"
        if decoded_latent is None:
            slices = [k for k in range(0, len(x), self.max_num_vae)]
            decoded_latent = []
            for s in slices:
                decoded_latent.append(self.decode_latent(x[s:s+self.max_num_vae]))
            decoded_latent = torch.cat(decoded_latent)

        if image_idx is None:
            image_path_list = self.config["image_path_list"]
        else:
            image_path_list = self.config["image_path_list"][image_idx]
        for img, img_path in zip(decoded_latent, image_path_list):
            if "image_id" in path:
                save_pth = path.replace("image_id", os.path.splitext(
                    os.path.basename(img_path))[0])
            else:
                save_pth = f'{path}/{os.path.splitext(os.path.basename(img_path))[0]}.png'
            save_dir = os.path.dirname(save_pth)
            os.makedirs(save_dir, exist_ok=True)
            T.ToPILImage()(img).save(save_pth)
        print("Image Saved to ", save_pth)
        return decoded_latent

    def save_images(self, images, image_path_list):
        for img, save_pth in zip(images, image_path_list):
            save_dir = os.path.dirname(save_pth)
            os.makedirs(save_dir, exist_ok=True)
            T.ToPILImage()(img).save(save_pth)
        # print("Image Saved to ", save_pth)

    def save_intermediate_x0(self, x, eps, t, image_idx=None):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        mu = alpha_prod_t ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        pred_x0 = (x - sigma * eps) / mu
        save_path = os.path.join(
            self.config["output_path"], "intermediates", f"{t}")
        self.decode_and_save(pred_x0, path=save_path, image_idx=image_idx)

    def warp_smooth(self, x0, t, fuse_ratio=0.5):
        decoded_latent = []
        slices = [k for k in range(0, len(x0), self.max_num_kf)]
        for s in slices:
            decoded_latent.append(self.decode_latent(x0[s:s+self.max_num_kf]))
        decoded_latent = torch.cat(decoded_latent)

        if self.save_intermediate:
            save_path = os.path.join(
                self.config["output_path"], "intermediates", f"{t}_before")
            self.decode_and_save(path=save_path, decoded_latent=decoded_latent)

        decoded_latent, masks = smooth_seq(
            decoded_latent, fuse_ratio=fuse_ratio)

        if self.save_intermediate:
            save_path = os.path.join(
                self.config["output_path"], "intermediates", f"{t}_after")
            self.decode_and_save(path=save_path, decoded_latent=decoded_latent)

        smoothed_x0 = []
        for s in slices:
            smoothed_x0.append(self.encode_imgs(
                decoded_latent[s:s+self.max_num_kf]))
        smoothed_x0 = torch.cat(smoothed_x0)
        masks = F.interpolate(masks, scale_factor=1 / 8).expand(-1, 4, -1, -1)
        masks = masks > 0.5
        smoothed_x0[~masks] = x0[~masks]
        return smoothed_x0

    def blur_smooth(self, x0, t, fuse_ratio=0.5):
        decoded_latent = []
        slices = [k for k in range(0, len(x0), self.max_num)]
        for s in slices:
            decoded_latent.append(self.decode_latent(x0[s:s+self.max_num]))
        decoded_latent = torch.cat(decoded_latent)

        if self.save_intermediate:
            save_path = os.path.join(
                self.config["output_path"], "intermediates", f"{t}_before")
            self.decode_and_save(path=save_path, decoded_latent=decoded_latent)

        blurred_latent = blur_video(decoded_latent)
        decoded_latent = fuse_ratio * blurred_latent + \
            (1 - fuse_ratio) * decoded_latent

        if self.save_intermediate:
            save_path = os.path.join(
                self.config["output_path"], "intermediates", f"{t}_after")
            self.decode_and_save(path=save_path, decoded_latent=decoded_latent)

        smoothed_x0 = []
        for s in slices:
            smoothed_x0.append(self.encode_imgs(
                decoded_latent[s:s+self.max_num]))
        smoothed_x0 = torch.cat(smoothed_x0)
        return smoothed_x0

    def pred_next_x(self, x, eps, t, i):
        timesteps = self.scheduler.timesteps
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[timesteps[i + 1]]
            if i < len(timesteps) - 1
            else self.scheduler.final_alpha_cumprod
        )
        mu = alpha_prod_t ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

        pred_x0 = (x - sigma * eps) / mu

        if self.use_warp_smooth and t >= self.min_warp_smooth and t <= self.max_warp_smooth:
            pred_x0 = self.warp_smooth(pred_x0, t)
            eps = (x - (pred_x0 * mu)) / sigma
        elif self.use_blur_smooth and t >= self.min_blur_smooth and t <= self.max_blur_smooth:
            pred_x0 = self.blur_smooth(pred_x0, t)
            eps = (x - (pred_x0 * mu)) / sigma
        elif self.save_intermediate:
            save_path = os.path.join(
                self.config["output_path"], "intermediates", "image_id", f"{t}.png")
            self.decode_and_save(x=pred_x0, path=save_path)

        
        
        eta = 0.0
        if eta != 0:
            timestep = t
            pred_epsilon = eps
            pred_original_sample = pred_x0
            variance_noise = randn_tensor(
                eps.shape, device=eps.device, dtype=eps.dtype
            )
            prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            variance = self.scheduler._get_variance(timestep, prev_timestep)
            std_dev_t = eta * variance ** (0.5)
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
            prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction + std_dev_t * variance_noise
            x = prev_sample
        else:
            x = mu_prev * pred_x0 + sigma_prev * eps
        return x


    def comptue_pca(self, feats):
        pcaxs = []

        for x in feats:
            fn = x.shape[0]
            N = x.shape[1]
            matx = rearrange(x, "(B F) N C -> B (F N) C", F = fn)[-1]
            mean = torch.mean(matx, dim=0, keepdim=True)
            matx = matx - mean
            q = min(256, matx.shape[0])
            U,S,V = torch.pca_lowrank(matx, q = q)
            pcax = torch.matmul(matx, V[:, :256])
            pcax = pcax / pcax.norm(dim=-1,keepdim=True)
            pcaxs += [pcax]
        
        pcax = torch.cat(pcaxs, dim = -1)
        U,S,V = torch.pca_lowrank(pcax, q = 4)
        pcax = torch.matmul(pcax, V[:, :3])
        # pcax = torch.cat([pcax[...,1:4], pcax[...,[0]]],dim=-1)
        # pca = sklearnPCA(n_components=4, svd_solver="arpack")
        # pcax = pca.fit_transform(pcax.cpu().numpy())
        # pcax = torch.from_numpy(pcax)
        h, w = self.eps.shape[-2:]
        ih, iw = self.image.shape[-2:]
        pcax = rearrange(pcax, "(B F h w) C -> (B F) C h w", F = fn, h = h, w = w)
        pcax = F.interpolate(pcax, size=(ih, iw), mode='bilinear')
        return pcax

    def after_loop(self, x, t):
        if self.save_pca and (t - 1) % 100 == 0:
            # pcas = tomesd.collect_from_patch(self.pipe, attr="pcax")
            # image_names = self.get_image_names()
            # for name, pcax in pcas.items():
            #     save_dir = os.path.join(
            #         self.config["output_path"], "pca", name)
            #     os.makedirs(save_dir, exist_ok=True)
            #     save_paths = [os.path.join(
            #         save_dir, f"{t}_{imn}.png") for imn in image_names]
            #     self.save_images(pcax[0], save_paths)

            # inter_xs = tomesd.collect_from_patch(self.pipe, attr="x")
            # image_names = self.get_image_names()
            # feats = []
            # for name, x in inter_xs.items():
            #     if "up_blocks" in name and "attentions.0" in name:

            #         feats += [x]
            # pcax = self.comptue_pca(feats)


            # save_dir = os.path.join(
            #     self.config["output_path"], "pca", "258feats")
            # os.makedirs(save_dir, exist_ok=True)
            # for ci in range(pcax.shape[1]):
            #     save_paths = [os.path.join(
            #         save_dir, f"{t}_c{ci}_{imn}.png") for imn in image_names]
            #     self.save_images(pcax[:,[ci],...], save_paths)
            # save_paths = [os.path.join(
            #     save_dir, f"{t}_alpha_{imn}.png") for imn in image_names]
            # self.save_images(pcax, save_paths)
            # save_paths = [os.path.join(
            #     save_dir, f"{t}_{imn}.png") for imn in image_names]
            # self.save_images(pcax[:,:3,...], save_paths)

            inter_xs = tomesd.collect_from_patch(self.pipe, attr="x")
            image_names = self.get_image_names()
            feats = []
            for name, x in inter_xs.items():
                if "up_blocks.3" in name:

                    feats += [x]
            pcax = self.comptue_pca(feats)


            save_dir = os.path.join(
                self.config["output_path"], "pca", "last3feats")
            os.makedirs(save_dir, exist_ok=True)
            for ci in range(pcax.shape[1]):
                save_paths = [os.path.join(
                    save_dir, f"{t}_c{ci}_{imn}.png") for imn in image_names]
                self.save_images(pcax[:,[ci],...], save_paths)
            save_paths = [os.path.join(
                save_dir, f"{t}_alpha_{imn}.png") for imn in image_names]
            self.save_images(pcax, save_paths)
            save_paths = [os.path.join(
                save_dir, f"{t}_{imn}.png") for imn in image_names]
            self.save_images(pcax[:,:3,...], save_paths)
        if self.save_tome and (t - 1) % 100 == 0:
            tomes = tomesd.collect_from_patch(self.pipe, attr="tome")
            image_names = self.get_image_names()
            N, C, H, W = self.source_image.shape
            flows = []
            for name, (m, u, h, w) in tomes.items():
                if "up_blocks" not in name:
                    continue
                cur_images = F.interpolate(self.source_image, (h, w))
                cur_tokens = rearrange(
                    cur_images, "(B F) C h w -> B (F h w) C", B=1)
                merged_tokens = m(cur_tokens, mode="replace", b_select=1)
                recon_tokens = u(merged_tokens, b_select=1, unm_modi="zero")
                recon_images = rearrange(
                    recon_tokens, "B (F h w) C -> (B F) C h w", h=h, w=w)
                recon_images = F.interpolate(recon_images, (H, W))
                save_dir = os.path.join(
                    self.config["output_path"], "tome", name)
                os.makedirs(save_dir, exist_ok=True)
                save_paths = [os.path.join(
                    save_dir, f"{t}_{imn}.png") for imn in image_names]
                self.save_images(recon_images, save_paths)

                grid_y, grid_x = torch.meshgrid(
                    torch.arange(0, h), torch.arange(0, w))
                # grids = torch.stack([grid_y / h, grid_x / w])
                grids = torch.stack([grid_y, grid_x])
                cur_coords = torch.zeros([N, 2, h, w]).to(x)
                cur_coords[:, 0:2, ...] = grids
                cur_tokens = rearrange(
                    cur_coords, "(B F) C h w -> B (F h w) C", B=1)
                merged_tokens = m(cur_tokens, mode="replace", b_select=-1)
                recon_tokens = u(merged_tokens, b_select=-1)
                recon_images = rearrange(
                    recon_tokens, "B (F h w) C -> (B F) C h w", h=h, w=w)
                coor_flow = (recon_images - cur_coords)

                
                coor_flow = torch.clamp(coor_flow, min=-5, max=5)
                coor_flow = flow_to_image(coor_flow)
                coor_flow = F.interpolate(coor_flow, (H, W), mode="nearest")
                # coor_flow = F.interpolate(coor_flow, size = (H,W), mode="bilinear")
                flows.append(coor_flow)
                save_dir = os.path.join(
                    self.config["output_path"], "tome", name)
                os.makedirs(save_dir, exist_ok=True)
                save_paths = [os.path.join(
                    save_dir, f"{t}_{imn}_coor.png") for imn in image_names]
                self.save_images(coor_flow, save_paths)
            flows = torch.stack(flows).to(torch.float).mean(dim = 0).to(torch.uint8)

            save_dir = os.path.join(
                    self.config["output_path"], "tome", "average")
            os.makedirs(save_dir, exist_ok=True)
            save_paths = [os.path.join(
                save_dir, f"{t}_{imn}_coor.png") for imn in image_names]
            self.save_images(flows, save_paths)
            
        if self.merge_global:
            tomesd.modify_patch(self.pipe, attr = "global_tokens", modi_value = None)

    def before_loop(self, x, t):
        if self.save:
            torch.save(x, os.path.join(
                self.save_path, f'noisy_latents_{t}.pt'))
        if "pnp" in self.control or "pnp" in self.control2:
            cur_latents = self.load_latent(t = t)
            self.cur_eps = cur_latents
        
        #Tmp
        # if t < 400:
        #     self.remove_tome()



    def sample_loop(self, x):
        flag = 0
        f = x.shape[0]
        # with torch.autocast(device_type='cuda', dtype=torch.float32):
        i = self.start_step
        for t in tqdm(self.scheduler.timesteps[self.start_step:], desc="Sampling"):
            self.before_loop(x, t)
            

            # if t < 500 and not flag:
            #     self.activate_tome()
            #     flag = 1

            # tomesd.update_patch(self.pipe, adhere_src=True, dst_frame=np.random.randint(0, F))
            # if t not in self.qk_injection_timesteps and not flag:
            #     print("Stop adhere src at T =", t.item())
            #     tomesd.update_patch(self.pipe, adhere_src=False)
            #     flag = 1

            # if self.config["cg"] != 0:
            #     x = self.denoise_step_cguidance(x, t)
            # else:
            # x = self.denoise_step(x, t)
            noise_pred = self.pred_noise(x, t)
            torch.cuda.empty_cache()
            # self.pipe.unet.to("cpu")
            # self.pipe.controlnet.to("cpu")
            x = self.pred_next_x(x, noise_pred, t, i)

            # self.pipe.unet.to("cuda")
            # self.pipe.controlnet.to("cuda")
            self.after_loop(x, t)
            i += 1

        self.pipe.unet = self.pipe.unet.to("cpu")
        self.pipe.vae = self.pipe.vae.to("cuda")
        if hasattr(self.pipe, "controlnet"):
            self.pipe.controlnet = self.pipe.controlnet.to("cpu")
        torch.cuda.empty_cache()
        decoded_latent = self.decode_and_save(x)
            # decoded_latent = self.decode_latent(x)
            # for img, img_path in zip(decoded_latent, self.config["image_path_list"]):
            #     save_pth = f'{self.config["output_path"]}/{os.path.splitext(os.path.basename(img_path))[0]}.png'
            #     T.ToPILImage()(img).save(save_pth)
            # print("Image Saved to ", save_pth)
        return decoded_latent

    def multi_frame_sample_loop(self, x):
        flag = 0
        f = x.shape[0]
        # x = x.to(torch.float16)
        noise_preds = torch.zeros_like(x)
        self.vae = self.vae.to("cpu")
        # with torch.autocast(device_type='cuda', dtype = x.dtype):


        #Whether load existing
        # res_path = f'{self.config["image_save_path"]}/tmp.x0'
        # if os.path.exists(res_path):
        #     self.pipe.unet = self.pipe.unet.to("cpu")
        #     if hasattr(self.pipe, "controlnet"):
        #         self.pipe.controlnet = self.pipe.controlnet.to("cpu")
        #     x = torch.load(res_path).to(x)
        #     self.vae = self.vae.to("cuda")
        #     decoded_latent = self.decode_and_save(x)
        #     return decoded_latent
        i = self.start_step
        for t in tqdm(self.scheduler.timesteps[self.start_step:], desc="Sampling"):
            # print("Max G-Memory", torch.cuda.max_memory_allocated(device=x.device))

            self.before_loop(x, t)
            # # tomesd.update_patch(self.pipe, ratio=0.8)
            # tomesd.update_patch(self.pipe, ratio = 0.6, coord_mask = False)
            # rand_arr = torch.randperm(f)
            # rand_arr = rand_arr[:self.kf_num]
            # select_mask = torch.zeros(f, dtype=torch.bool)
            # select_mask[rand_arr] = 1
            # slices = [k for k in range(0, len(rand_arr), self.max_num_kf)]
            # for s in slices:
            #     cur_idx = rand_arr[s:s+self.max_num_kf]
            #     # x[cur_idx] = self.denoise_step(x[cur_idx], t, image=self.image[cur_idx], image_idx = cur_idx)
            #     noise_preds[cur_idx] = self.pred_noise(
            #         x[cur_idx], t, image=self.image[cur_idx])

            # tomesd.update_patch(self.pipe, ratio = 0.75, coord_mask = False)
            # rand_start = torch.randint(0, f, size=torch.Size([1]))
            # rand_arr = torch.arange(f)
            # rand_arr = torch.cat(
            #     [rand_arr[rand_start:], rand_arr[:rand_start]])[::self.kf_interval]
            rand_arr = torch.randperm(f)
            rand_arr = rand_arr[:self.kf_num]
            select_mask = torch.zeros(f, dtype=torch.bool)
            select_mask[rand_arr] = 1
            slices = [k for k in range(0, len(rand_arr), self.max_num_kf)]
            for s in slices:
                cur_idx = rand_arr[s:s+self.max_num_kf]
                # x[cur_idx] = self.denoise_step(x[cur_idx], t, image=self.image[cur_idx], image_idx = cur_idx)
                noise_preds[cur_idx] = self.pred_noise(
                    x[cur_idx], t, image_idx=cur_idx)


            # tomesd.update_patch(self.pipe, ratio=0.8)
            # tomesd.update_patch(self.pipe, ratio = 0.9, coord_mask = self.coord_mask)
            inter_idx = torch.arange(f)[~select_mask]

            # inter_x = x[~select_mask]
            # inter_images = self.image[~select_mask]
            inter_F = len(inter_idx)

            #Type 1
            # rand_start = torch.randint(0, inter_F, size=torch.Size([1]))
            # order = torch.arange(inter_F)
            # rand_order = torch.cat(
            #     [order[rand_start:], order[:rand_start]])

            # slices = [k for k in range(0, inter_F, self.max_num_int)]
            # for s in slices:
            #     cur_idx = inter_idx[rand_order[s:s+self.max_num_int]]
            #     # x[cur_idx] = self.denoise_step(x[cur_idx], t, image=self.image[cur_idx], image_idx = cur_idx)
            #     noise_preds[cur_idx] = self.pred_noise(
            #         x[cur_idx], t, image_idx=cur_idx)


            #Type 2
            cat_flag = True
            starts, ends = self.get_rand_split(inter_F)
            for j, (s, e) in enumerate(zip(starts, ends)):
                torch.cuda.empty_cache()
                cur_idx = inter_idx[s:e]
                # if len(cur_idx) < self.max_num_kf:
                #     if not cat_flag:
                #         continue
                #     assert s == 0 or e == inter_F
                #     if s == 0:
                #         ns, ne = inter_F - (self.max_num_kf - (e - s)), inter_F
                #     else:
                #         ns, ne = 0, (self.max_num_kf - (e - s))
                #     cur_idx = torch.cat([cur_idx, inter_idx[ns:ne]])
                #     cat_flag = False
                # x[cur_idx] = self.denoise_step(x[cur_idx], t, image=self.image[cur_idx], image_idx = cur_idx)
                noise_preds[cur_idx] = self.pred_noise(
                    x[cur_idx], t, image_idx=cur_idx)
        
            # x[~select_mask] = inter_x
            # self.pipe.unet.to("cpu")
            # self.pipe.controlnet.to("cpu")
            # torch.cuda.empty_cache()
            x = self.pred_next_x(x, noise_preds, t, i)


            self.after_loop(x, t)
            # self.pipe.unet.to("cuda")
            # self.pipe.controlnet.to("cuda")
            # rand_start = torch.randint(0, f, size = torch.Size([1]))
            # order = torch.arange(f)
            # rand_order = torch.cat([order[rand_start:], order[:rand_start]])
            # slices = [k for k in range(0, len(rand_order), self.max_num)]
            # for s in slices:
            #    x[rand_order[s:s+self.max_num]] = self.denoise_step(x[rand_order[s:s+self.max_num]], t, image=self.image[rand_order[s:s+self.max_num]])
            i += 1
        self.pipe.unet = self.pipe.unet.to("cpu")
        if hasattr(self.pipe, "controlnet"):
            self.pipe.controlnet = self.pipe.controlnet.to("cpu")
        self.vae = self.vae.to("cuda")
        torch.cuda.empty_cache()
        torch.save(x, f'{self.config["output_path"]}/tmp.x0')
        decoded_latent = self.decode_and_save(x)
        return decoded_latent




    def get_rand_split(self, inter_F):
        rand_start = torch.randint(1, self.max_num_kf + 1, torch.Size([1]))
        # print(rand_start)
        slices = [0] + [k for k in range(rand_start, inter_F + 1, self.max_num_int)]
        if slices[-1] != inter_F:
            slices.append(inter_F)
        
        # rand_direction = torch.randint(2, torch.Size([1]))
        rand_direction = torch.randint(2, torch.Size([1]))
        # rand_direction = 1
        if rand_direction:
            starts = slices[:-1]
            ends = slices[1:]
        else:
            starts = slices[-2::-1]
            ends = slices[:0:-1]

        if self.chunk_ord == "perm":
            randord = torch.randperm(len(starts))
            starts = [starts[i] for i in randord]
            ends = [ends[i] for i in randord]
        # starts = starts[randord[0]:] + starts[:randord[0]]
        # ends = ends[randord[0]:] + ends[:randord[0]]
        elif self.chunk_ord == "mix":
            if hasattr(self, "permdiv"):
                permdiv = self.permdiv
            else:
                permdiv = 3.
            
            randord = torch.randperm(len(starts)).tolist()
            rand_len = int(len(randord) / permdiv)
            seqord = sorted(randord[rand_len:])
            randord = randord[:rand_len]
            if abs(seqord[-1] - randord[-1]) < abs(seqord[0] - randord[-1]):
                seqord = seqord[::-1]
            cord = randord + seqord
            # print(cord)
            starts = [starts[i] for i in cord]
            ends = [ends[i] for i in cord]
        return starts,ends

def process_config(opt, config):


    config["multi_frames"] = opt.multi_frames
    opt.frame_range = (int(opt.frame_range[0]), int(opt.frame_range[1]))
    if opt.multi_frames:
        suffix = f"{opt.frame_range[0]}-{opt.frame_range[1]}"
        opt.imgids = list(range(opt.frame_range[0], opt.frame_range[1] + 1))
    else:
        suffix = "{:04}".format(int(opt.imgids[0]))
        for imgid in opt.imgids[1:]:
            suffix += f"&{imgid}"
    config["suffix"] = suffix
    # config["prev_edit_image_path"] = "PNP-results/0002/pre.png"
    control_suffix = opt.control + opt.multi_control if opt.multi_control != "none" else opt.control
    config["output_path"] = os.path.join(config["output_path"], suffix, control_suffix)

    config["subdir"] = opt.subdir
    config["image_save_path"] = os.path.join(config["output_path"], opt.subdir)


    config["image_path_list"] = []
    # endls = [".png", ".jpg"]
    # for imgid in opt.imgids:
    #     csuffix = "{:04}".format(int(imgid))
    #     p = 0
    #     pth = os.path.join(config["image_path"], csuffix + endls[p])
    #     while not os.path.exists(pth) and p < len(endls):
    #         p += 1
    #         pth = os.path.join(config["image_path"], csuffix + endls[p])

    #     assert os.path.exists(pth), f"Image {csuffix} not found"
    #     config["image_path_list"].append(pth)
    img_paths = glob.glob(os.path.join(config["image_path"], "*.png"))
    img_paths += glob.glob(os.path.join(config["image_path"], "*.jpg"))
    img_paths = sorted(img_paths)
    img_indexs = []
    # if not opt.multi_frames:
    for i, img_pth in enumerate(img_paths):
        idx = int(os.path.basename(img_pth).split(".")[0])
        if idx in opt.imgids:
            config["image_path_list"].append(img_pth)
            img_indexs += [i]
    # else:
    #     config["image_path_list"] = img_paths
    #     img_indexs = [i for i in range(len(img_paths))]
    config["img_indexes"] = img_indexs
    
    if "negative_prompt" not in config or config["negative_prompt"] == None:
        config["negative_prompt"] = ""

    # if len(opt.prev_imgids) > 0:
    #     csuffix = "{:04}".format(int(opt.prev_imgids[0]))
    #     for imgid in opt.prev_imgids[1:]:
    #         csuffix += f"&{int(imgid)}"
    #     config["cond_path"] = os.path.join(
    #         config["latents_path"] + "G", csuffix)
    config["save"] = opt.save
    config["control"] = opt.control

    os.makedirs(config["output_path"], exist_ok=True)
    with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # config["cond_pos"] = torch.tensor(opt.cond_pos, dtype=torch.long)
    # config["prev_cond_pos"] = torch.tensor(opt.prev_cond_pos, dtype=torch.long)
    config["image_path_list"] = np.array(config["image_path_list"])
    config["multi_control"] = opt.multi_control
    seed_everything(config["seed"])
    return config


# Continuous Warp
if __name__ == '__main__':
    # torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='pnp-configs/examples/config_breakdance.yaml')
# hon pnp_tile.py --config_path pnp-configs/vidtome/config_blackswan.yaml --control pnp-sddepth --multi_frames --frame_range 0 31
    # parser.add_argument('--imgids', nargs='+', default=[1, 9, 17, 25, 33, 41, 49, 57])
    parser.add_argument('--imgids', nargs='+', default=[0])
    parser.add_argument('--subdir', type=str, default="")
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--prev_imgids', nargs='+', default=[])
    parser.add_argument('--cond_pos', nargs='+', default=[0, 1])
    parser.add_argument('--prev_cond_pos', nargs='+', default=[-8, -1])
    parser.add_argument('--multi_frames', action="store_true")
    parser.add_argument('--frame_range', nargs='+', default=[0, 31])
    parser.add_argument('--control', type=str, default="none",
                        choices=["tile", "ip2p", "openpose", "softedge", "depth", "none", "lineart_anime", "canny", "pnp", "depth_dinv", "softedge_sinv", "pnp-sddepth", "pnp-sddepth2"])
    parser.add_argument('--multi_control', type=str, default="none",
                        choices=["tile", "ip2p", "openpose", "softedge", "depth", "none", "lineart_anime", "canny", "pnp"])   
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)

    config = process_config(opt, config)
    print(config)
    pnp = PNP(config)
    pnp.run_pnp()

    # pnp.save_noise()
