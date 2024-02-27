import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import os
from transformers import logging

from utils import CONTROLNET_DICT
from utils import load_config, save_config
from utils import get_controlnet_kwargs, get_frame_ids, get_latents_dir, init_model, seed_everything
from utils import prepare_control, load_latent, load_video, prepare_depth, save_video
from utils import register_time, register_attention_control, register_conv_control

import vidtome

# suppress partial model loading warning
logging.set_verbosity_error()


class Generator(nn.Module):
    def __init__(self, pipe, scheduler, config):
        super().__init__()

        self.device = config.device
        self.seed = config.seed



        
        self.model_key = config.model_key

        self.config = config
        gene_config = config.generation
        float_precision = gene_config.float_precision if "float_precision" in gene_config else config.float_precision
        if float_precision == "fp16":
            self.dtype = torch.float16
            print("[INFO] float precision fp16. Use torch.float16.")
        else:
            self.dtype = torch.float32
            print("[INFO] float precision fp32. Use torch.float32.")

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        if config.enable_xformers_memory_efficient_attention:
            pipe.enable_xformers_memory_efficient_attention()
        self.n_timesteps = gene_config.n_timesteps
        scheduler.set_timesteps(gene_config.n_timesteps, device=self.device)
        self.scheduler = scheduler

        self.batch_size = 2
        self.control = gene_config.control
        self.use_depth = config.sd_version == "depth"
        self.use_controlnet = self.control in CONTROLNET_DICT.keys()
        self.use_pnp = self.control == "pnp"
        if self.use_controlnet:
            self.controlnet = pipe.controlnet
            self.controlnet_scale = gene_config.control_scale
        elif self.use_pnp:
            pnp_f_t = int(gene_config.n_timesteps * gene_config.pnp_f_t)
            pnp_attn_t = int(gene_config.n_timesteps * gene_config.pnp_attn_t)
            self.batch_size += 1
            self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)

        self.chunk_size = gene_config.chunk_size
        self.chunk_ord = gene_config.chunk_ord
        self.merge_global = gene_config.merge_global
        self.local_merge_ratio = gene_config.local_merge_ratio
        self.global_merge_ratio = gene_config.global_merge_ratio
        self.global_rand = gene_config.global_rand
        self.align_batch = gene_config.align_batch

        self.prompt = gene_config.prompt
        self.negative_prompt = gene_config.negative_prompt
        self.guidance_scale = gene_config.guidance_scale
        self.save_frame = gene_config.save_frame

        self.frame_height, self.frame_width = config.height, config.width
        self.work_dir = config.work_dir

        self.chunk_ord = gene_config.chunk_ord
        if "mix" in self.chunk_ord:
            self.perm_div = float(self.chunk_ord.split("-")[-1]) if "-" in self.chunk_ord else 3.
            self.chunk_ord = "mix"
        # Patch VidToMe to model
        self.activate_vidtome()

        if gene_config.use_lora:
            self.pipe.load_lora_weights(**gene_config.lora)
    
    def activate_vidtome(self):
        vidtome.apply_patch(self.pipe, self.local_merge_ratio, self.merge_global, self.global_merge_ratio, 
            seed = self.seed, batch_size = self.batch_size, align_batch = self.use_pnp or self.align_batch, global_rand = self.global_rand)        

    @torch.no_grad()
    def get_text_embeds_input(self, prompt, negative_prompt):
        text_embeds = self.get_text_embeds(
            prompt, negative_prompt, self.device)
        if self.use_pnp:
            pnp_guidance_embeds = self.get_text_embeds("", device=self.device)
            text_embeds = torch.cat(
                [pnp_guidance_embeds, text_embeds], dim=0)
        return text_embeds

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt=None, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        if negative_prompt is not None:
            uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                          return_tensors='pt')
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def prepare_data(self, data_path, latent_path, frame_ids):
        self.frames = load_video(data_path, self.frame_height,
                                 self.frame_width, frame_ids=frame_ids, device=self.device)
        self.init_noise = load_latent(
            latent_path, t=self.scheduler.timesteps[0], frame_ids=frame_ids).to(self.dtype).to(self.device)

        if self.use_depth:
            self.depths = prepare_depth(
                self.pipe, self.frames, frame_ids, self.work_dir).to(self.init_noise)

        if self.use_controlnet:
            self.controlnet_images = prepare_control(
                self.control, self.frames, frame_ids, self.work_dir).to(self.init_noise)

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def decode_latents_batch(self, latents):
        imgs = []
        batch_latents = latents.split(self.batch_size, dim=0)
        for latent in batch_latents:
            imgs += [self.decode_latents(latent)]
        imgs = torch.cat(imgs)
        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def encode_imgs_batch(self, imgs):
        latents = []
        batch_imgs = imgs.split(self.batch_size, dim=0)
        for img in batch_imgs:
            latents += [self.encode_imgs(img)]
        latents = torch.cat(latents)
        return latents
    
    def get_chunks(self, flen):
        x_index = torch.arange(flen)

        # The first chunk has a random length
        rand_first = np.random.randint(0, self.chunk_size) + 1
        chunks = x_index[rand_first:].split(self.chunk_size, dim=0)
        chunks = [x_index[:rand_first]] + list(chunks)
        if np.random.rand() > 0.5:
            chunks = chunks[::-1]
        
        # Chunk order only matter when we do global token merging
        if self.merge_global == False:
            return chunks

        # Chunk order. "seq": sequential order. "rand": full permutation. "mix": partial permutation.
        if self.chunk_ord == "rand":
            order = torch.randperm(len(chunks))
        elif self.chunk_ord == "mix":
            randord = torch.randperm(len(chunks)).tolist()
            rand_len = int(len(randord) / self.perm_div)
            seqord = sorted(randord[rand_len:])
            randord = randord[:rand_len]
            if abs(seqord[-1] - randord[-1]) < abs(seqord[0] - randord[-1]):
                seqord = seqord[::-1]
            order = randord + seqord
        else:
            order = torch.arange(len(chunks))
        chunks = [chunks[i] for i in order]
        return chunks

    @torch.no_grad()
    def ddim_sample(self, x, conds):
        print("[INFO] denoising frames...")
        timesteps = self.scheduler.timesteps
        noises = torch.zeros_like(x)

        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            self.pre_iter(x, t)

            # Split video into chunks and denoise
            chunks = self.get_chunks(len(x))
            for chunk in chunks:
                torch.cuda.empty_cache()
                noises[chunk] = self.pred_noise(
                    x[chunk], conds, t, batch_idx=chunk)

            x = self.pred_next_x(x, noises, t, i, inversion=False)

            self.post_iter(x, t)
        return x

    def pre_iter(self, x, t):
        if self.use_pnp:
            # Prepare PnP
            register_time(self, t.item())
            cur_latents = load_latent(self.latent_path, t=t, frame_ids = self.frame_ids)
            self.cur_latents = cur_latents

    def post_iter(self, x, t):
        if self.merge_global:
            # Reset global tokens
            vidtome.update_patch(self.pipe, global_tokens = None)

    @torch.no_grad()
    def pred_noise(self, x, cond, t, batch_idx=None):

        flen = len(x)
        text_embed_input = cond.repeat_interleave(flen, dim=0)

        # For classifier-free guidance
        latent_model_input = torch.cat([x, x])
        batch_size = 2

        if self.use_pnp:
            # Cat latents from inverted source frames for PnP operation
            source_latents = self.cur_latents
            if batch_idx is not None:
                source_latents = source_latents[batch_idx]
            latent_model_input = torch.cat([source_latents.to(x), latent_model_input])
            batch_size += 1

        # For sd-depth model
        if self.use_depth:
            depth = self.depths
            if batch_idx is not None:
                depth = depth[batch_idx]
            depth = depth.repeat(batch_size, 1, 1, 1)
            latent_model_input = torch.cat([latent_model_input, depth.to(x)], dim=1)
        
        kwargs = dict()
        # Compute controlnet outputs
        if self.use_controlnet:
            controlnet_cond = self.controlnet_images
            if batch_idx is not None:
                controlnet_cond = controlnet_cond[batch_idx]
            controlnet_cond = controlnet_cond.repeat(batch_size, 1, 1, 1)
            controlnet_kwargs = get_controlnet_kwargs(
                self.controlnet, latent_model_input, text_embed_input, t, controlnet_cond, self.controlnet_scale)
            kwargs.update(controlnet_kwargs)
        # Pred noise!
        eps = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input, **kwargs).sample
        noise_pred_uncond, noise_pred_cond = eps.chunk(batch_size)[-2:]
        # CFG
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        return noise_pred

    @torch.no_grad()
    def pred_next_x(self, x, eps, t, i, inversion=False):
        if inversion:
            timesteps = reversed(self.scheduler.timesteps)
        else:
            timesteps = self.scheduler.timesteps
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        if inversion:
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else self.scheduler.final_alpha_cumprod
            )
        else:
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else self.scheduler.final_alpha_cumprod
            )
        mu = alpha_prod_t ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

        if inversion:
            pred_x0 = (x - sigma_prev * eps) / mu_prev
            x = mu * pred_x0 + sigma * eps
        else:
            pred_x0 = (x - sigma * eps) / mu
            x = mu_prev * pred_x0 + sigma_prev * eps

        return x

    def init_pnp(self, conv_injection_t, qk_injection_t):
        qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control(
            self, qk_injection_timesteps, num_inputs=self.batch_size)
        register_conv_control(
            self, conv_injection_timesteps, num_inputs=self.batch_size)

    def check_latent_exists(self, latent_path):
        if self.use_pnp:
            timesteps = self.scheduler.timesteps
        else:
            timesteps = [self.scheduler.timesteps[0]]

        for ts in timesteps:
            cur_latent_path = os.path.join(
                latent_path, f'noisy_latents_{ts}.pt')
            if not os.path.exists(cur_latent_path):
                return False
        return True

    @torch.no_grad()
    def __call__(self, data_path, latent_path, output_path, frame_ids):
        self.scheduler.set_timesteps(self.n_timesteps)
        latent_path = get_latents_dir(latent_path, self.model_key)
        assert self.check_latent_exists(
            latent_path), f"Required latent not found at {latent_path}. \
                    Note: If using PnP as control, you need inversion latents saved \
                     at each generation timestep."
        
        self.data_path = data_path
        self.latent_path = latent_path
        self.frame_ids = frame_ids
        self.prepare_data(data_path, latent_path, frame_ids)

        print(f"[INFO] initial noise latent shape: {self.init_noise.shape}")

        for edit_name, edit_prompt in self.prompt.items():
            print(f"[INFO] current prompt: {edit_prompt}")
            conds = self.get_text_embeds_input(edit_prompt, self.negative_prompt)
            # Comment this if you have enough GPU memory
            clean_latent = self.ddim_sample(self.init_noise, conds)
            torch.cuda.empty_cache()
            clean_frames = self.decode_latents_batch(clean_latent)
            cur_output_path = os.path.join(output_path, edit_name)
            save_config(self.config, cur_output_path, gene = True)
            save_video(clean_frames, cur_output_path, save_frame = self.save_frame)


        


if __name__ == "__main__":
    config = load_config()
    pipe, scheduler, model_key = init_model(
        config.device, config.sd_version, config.model_key, config.generation.control, config.float_precision)
    config.model_key = model_key
    seed_everything(config.seed)
    generator = Generator(pipe, scheduler, config)
    frame_ids = get_frame_ids(
        config.generation.frame_range, config.generation.frame_ids)
    generator(config.input_path, config.generation.latents_path,
              config.generation.output_path, frame_ids=frame_ids)
