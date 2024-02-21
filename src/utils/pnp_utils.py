import torch
import os
import random
import numpy as np
import torch.nn.functional as F
from einops import rearrange


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def register_time(model, t):
    # conv_module = model.unet.up_blocks[1].resnets[1]
    # setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            if hasattr(model.unet.up_blocks[res], "attentions"):
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                setattr(module, 't', t)
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
                setattr(module, 't', t)
            conv_module = model.unet.up_blocks[res].resnets[block]
            setattr(conv_module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            if hasattr(model.unet.down_blocks[res], "attentions"):
                module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
                setattr(module, 't', t)
                module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
                setattr(module, 't', t)
            conv_module = model.unet.down_blocks[res].resnets[block]
            setattr(conv_module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 't', t)


def register_flow(model, flow, occlusion_mask = None):
    # conv_module = model.unet.up_blocks[1].resnets[1]
    # setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'flow', flow)
            setattr(module, 'occlusion_mask', occlusion_mask)
            conv_module = model.unet.up_blocks[res].resnets[block]
            setattr(conv_module, 'flow', flow)
            setattr(conv_module, 'occlusion_mask', occlusion_mask)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            if hasattr(model.unet.down_blocks[res], "attentions"):
                module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
                setattr(module, 'flow', flow)
                setattr(module, 'occlusion_mask', occlusion_mask)
            conv_module = model.unet.down_blocks[res].resnets[block]
            setattr(conv_module, 'flow', flow)
            setattr(conv_module, 'occlusion_mask', occlusion_mask)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', flow)


def load_source_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
    assert os.path.exists(
        latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
    latents = torch.load(latents_t_path)
    return latents


def register_attention_control_efficient(model, injection_schedule, num_inputs):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // num_inputs)
                # inject unconditional
                # q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                # k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                # # q[source_batch_size:2 * source_batch_size] = q[3 * source_batch_size:4 * source_batch_size]
                # # k[source_batch_size:2 * source_batch_size] = k[3 * source_batch_size:4 * source_batch_size]
                # # inject conditional
                # if num_inputs > 2:
                #     q[2 * source_batch_size:3 * source_batch_size] = q[:source_batch_size]
                #     k[2 * source_batch_size:3 * source_batch_size] = k[:source_batch_size]

                q = q[:source_batch_size]
                k = k[:source_batch_size]
                # q[2 * source_batch_size:3 * source_batch_size] = q[3 * source_batch_size: 4 * source_batch_size]
                # k[2 * source_batch_size:3 * source_batch_size] = k[3 * source_batch_size: 4 * source_batch_size]
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)



            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)


            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                attn = torch.cat([attn] * num_inputs, dim = 0)
                # attn = attn.repeat(num_inputs, 1, 1)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)
    print("Register Source Attention QK Injection in Up Res", res_dict)


def register_conv_control_efficient(model, injection_schedule, num_inputs):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[
                    :, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // num_inputs)
                # inject unconditional
                hidden_states[source_batch_size:2 *
                              source_batch_size] = hidden_states[:source_batch_size]
                # hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[3 * source_batch_size:4 * source_batch_size]
                # inject conditional
                if num_inputs > 2:
                    hidden_states[2 * source_batch_size:3 *
                                source_batch_size] = hidden_states[:source_batch_size]
                # hidden_states[2 * source_batch_size:3 * source_batch_size] = hidden_states[3 * source_batch_size:4 * source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / \
                self.output_scale_factor

            return output_tensor

        return forward
    res_dict = {1: [1]}
    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)
    print("Register Source Feature Injection in Up Res", res_dict)

def register_prev_control_efficient(model, injection_schedule, num_inputs):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[
                    :, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // num_inputs)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[3 *
                                                                                       source_batch_size:4 * source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:3 * source_batch_size] = hidden_states[3 *
                                                                                           source_batch_size:4 * source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / \
                self.output_scale_factor

            return output_tensor

        return forward
    # res_dict = {1: [2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    # res_dict = {1: [1, 2], 2: [0, 1, 2]}
    res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1]}
    for res in res_dict:
        for block in res_dict[res]:
            # module = model.unet.up_blocks[res].resnets[block]
            module = model.unet.down_blocks[res].resnets[block]
            module.forward = conv_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)
    # conv_module = model.unet.up_blocks[1].resnets[1]
    # conv_module.forward = conv_forward(conv_module)
    # setattr(conv_module, 'injection_schedule', injection_schedule)
    
def register_prev_control_efficient_att(model, injection_schedule, num_inputs):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // num_inputs)
                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q[3 *
                                                               source_batch_size: 4 * source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[3 *
                                                               source_batch_size: 4 * source_batch_size]
                # inject conditional
                q[2 * source_batch_size:3 * source_batch_size] = q[3 *
                                                                   source_batch_size: 4 * source_batch_size]
                k[2 * source_batch_size:3 * source_batch_size] = k[3 *
                                                                   source_batch_size: 4 * source_batch_size]

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    res_dict = {1: [0, 1], 2: [0, 1], 0: [0, 1]}
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)


def register_prev_control_efficient_attmap_warp(model, injection_schedule, num_inputs):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)

            v = self.to_v(encoder_hidden_states)

            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(v.shape[0] // num_inputs)

                # nv = v.clone()
                # nk = k.clone()
                # nv[source_batch_size:2 * source_batch_size] = v[3 * source_batch_size: 4 * source_batch_size]
                # nk[source_batch_size:2 * source_batch_size] = k[3 * source_batch_size: 4 * source_batch_size]
                # nv[2 * source_batch_size:3 * source_batch_size] = v[3 * source_batch_size: 4 * source_batch_size]
                # nk[2 * source_batch_size:3 * source_batch_size] = k[3 * source_batch_size: 4 * source_batch_size]

                # v = torch.cat([v, nv], dim=1)
                # k = torch.cat([k, nk], dim=1)

                v[source_batch_size:2 * source_batch_size] = v[3 *
                                                               source_batch_size: 4 * source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[3 *
                                                               source_batch_size: 4 * source_batch_size]
                # q[2 * source_batch_size:3 * source_batch_size] = q[:source_batch_size]
                # k[2 * source_batch_size:3 * source_batch_size] = k[:source_batch_size]
                v[2 * source_batch_size:3 * source_batch_size] = v[3 *
                                                                   source_batch_size: 4 * source_batch_size]
                k[2 * source_batch_size:3 * source_batch_size] = k[3 *
                                                                   source_batch_size: 4 * source_batch_size]

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # if not is_cross and self.injection_schedule is not None and (
            #         self.t in self.injection_schedule or self.t == 1000):
            #     source_batch_size = int(sim.shape[0] // num_inputs)
            #     prev_att = sim[3 * source_batch_size: 4 * source_batch_size]
            #     prev_att_n = sim[4 * source_batch_size: 5 * source_batch_size]

            #     size = int(np.sqrt(prev_att.shape[2]))
            #     prev_att = prev_att.reshape(*prev_att.shape[:2], size, size)
            #     prev_att_n = prev_att_n.reshape(*prev_att.shape[:2], size, size)
            #     # scale_factor = size / self.flow.shape[0]
            #     # flow_downsampled = F.interpolate(switch_format(self.flow), scale_factor=scale_factor, mode='bilinear', align_corners=True) * scale_factor
            #     warped_sim = warp_frame_tensor(self.flow, prev_att, mode = "bilinear")
            #     warped_sim = warped_sim.reshape(*warped_sim.shape[:2], -1)
            #     warped_sim_n = warp_frame_tensor(self.flow, prev_att_n, mode = "bilinear")
            #     warped_sim_n = warped_sim_n.reshape(*warped_sim_n.shape[:2], -1)
            #     sim[2 * source_batch_size: 3 * source_batch_size] = warped_sim
            #     # sim[2 * source_batch_size: 3 * source_batch_size] = prev_att
            #     sim[source_batch_size: 2 * source_batch_size] = warped_sim
            #     # sim[source_batch_size: 2 * source_batch_size] = prev_att

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward
    # res_dict = {1: [0, 1], 2: [0, 1], 0: [0, 1]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    # for res in res_dict:
    #     for block in res_dict[res]:
    #         module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
    #         module.forward = sa_forward(module)
    #         setattr(module, 'injection_schedule', injection_schedule)
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)
    print("Register Prev Attention Map Injection in Up Res", res_dict)


def register_cross_attn_store_efficient(model, save_dir, latent, save_timesteps):
    def sa_forward(self, l_shape, save_timesteps):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            if self.t in save_timesteps:
                save_path = self.save_path.replace("Timestep", f"{self.t}")
                torch.save(rearrange(attn, "b (h w) c -> b h w c",
                           h=l_shape[0]), save_path)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward
    latent_hw_dict = {0: (latent.shape[2], latent.shape[3])}
    for i in range(1, 4):
        latent_hw_dict[i] = (int(np.ceil(latent_hw_dict[i - 1][0] / 2)),
                             int(np.ceil(latent_hw_dict[i - 1][1] // 2)))
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            module.forward = sa_forward(
                module, latent_hw_dict[res], save_timesteps)
            setattr(module, 'save_path', os.path.join(
                save_dir, "Timestep", f"downblock{res}_id{block}.att"))
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            module.forward = sa_forward(
                module, latent_hw_dict[3 - res], save_timesteps)
            setattr(module, 'save_path', os.path.join(
                save_dir, "Timestep", f"upblock{res}_id{block}.att"))


def register_cross_attn_p2p_efficient(model,injection_schedule, num_inputs):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(v.shape[0] // num_inputs)
                attn[2 * source_batch_size:3 * source_batch_size] = attn[:source_batch_size]
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)
    print("Register Cross Attention Map P2P in Up Res", res_dict)


    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)
    print("Register Cross Attention Map P2P in Up Res", res_dict)
