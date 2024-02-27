import torch
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode: str = None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

# For Local Token Merging
def bipartite_soft_matching_randframe(metric: torch.Tensor, 
                                      F: int, ratio: float, unm_pre: int, generator: torch.Generator,
                                      target_stride: int = 4, align_batch: bool = False,
                                      merge_mode: str = "replace") -> Tuple[Callable, Callable, dict]:
    """
    Partitions the multi-frame tokens into src and dst and merges ratio of src tokens from src to dst.
    Dst tokens are partitioned by choosing one random frame.

    Args:
        - metric [B, N, C]: metric to use for similarity.
        - F: frame number.
        - ratio: ratio of src tokens to be removed (by merging).
        - unm_pre: number of src tokens not merged at previous ToMe. Pre-sequence: [unm_pre|F_0|F_1|...]
        - generator: random number generator
        - target_stride: stride of target frame.
        - align_batch: whether to align similarity matching maps of samples in the batch. True when using PnP.
        - merge_mode: how to merge tokens. "mean": tokens -> Mean(src_token, dst_token); "replace": tokens -> dst_token.

    Returns:
        Merge and unmerge operation according to the matching result. Return a dict including other values.
    """
    B, N, _ = metric.shape
    # Compute pre-frame token number. N = unm_pre + tnum * F.
    tnum = (N - unm_pre) // F

    if ratio <= 0:
        return do_nothing, do_nothing, {"unm_num": tnum}

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        # Prepare idx buffer. Ignore previous unmerged tokens.
        idx_buffer = torch.arange(
            N - unm_pre, device=metric.device, dtype=torch.int64)

        # Select the random target frame.
        target_stride = min(target_stride, F)
        randf = torch.randint(0, target_stride, torch.Size(
            [1]), generator=generator, device=generator.device)
        dst_select = ((torch.div(idx_buffer, tnum, rounding_mode='floor')) %
                      target_stride == randf).to(torch.bool)

        # a_idx: src index. b_idx: dst index
        a_idx = idx_buffer[None, ~dst_select, None] + unm_pre
        b_idx = idx_buffer[None, dst_select, None] + unm_pre

        # Add unmerged tokens to dst.
        unm_buffer = torch.arange(unm_pre, device=metric.device, dtype=torch.int64)[
            None, :, None]
        b_idx = torch.cat([b_idx, unm_buffer], dim=1)

        # We're finished with these
        del idx_buffer, unm_buffer

        num_dst = b_idx.shape[1]

        def split(x):
            # Split src, dst tokens
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(x, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst

        # Cosine similarity between src and dst tokens
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)

        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], int(a.shape[1] * ratio))


        if align_batch:
            # Cat scores of all samples in the batch. When using PnP, samples are (src, neg, pos).
            # Find the most similar greedily among all samples.
            scores = torch.cat([*scores], dim=-1)
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None],
                             dim=-2, index=src_idx) % num_dst # Map index to (0, num_dst - 1)
            
            # Use the same matching result for all samples
            unm_idx = unm_idx.expand(B, -1, -1)
            src_idx = src_idx.expand(B, -1, -1)
            dst_idx = dst_idx.expand(B, -1, -1)
        else:

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode=None) -> torch.Tensor:
        # Merge tokens according to matching result.
        src, dst = split(x)
        n, t1, c = src.shape
        u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx

        unm = gather(src, dim=-2, index=u_idx.expand(-1, -1, c))
        mode = mode if mode is not None else merge_mode
        if mode != "replace":
            src = gather(src, dim=-2, index=s_idx.expand(-1, -1, c))
            # In other mode such as mean, combine matched src and dst tokens.
            dst = dst.scatter_reduce(-2, d_idx.expand(-1, -1, c),
                                     src, reduce=mode, include_self=True)
        # In replace mode, just cat unmerged tokens and dst tokens. Ignore src tokens.
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor, **kwarg) -> torch.Tensor:
        # Unmerge tokens to original size according to matching result.
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        b, _, c = unm.shape
        u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx
        # Restored src tokens take value from dst tokens
        src = gather(dst, dim=-2, index=d_idx.expand(-1, -1, c))

        # Combine back to the original shape
        out = torch.zeros(b, N, c, device=x.device, dtype=x.dtype)
        # Scatter dst tokens
        out.scatter_(dim=-2, index=b_idx.expand(b, -1, c), src=dst)
        # Scatter unmerged tokens
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1),
                     dim=1, index=u_idx).expand(-1, -1, c), src=unm)
        # Scatter src tokens
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1),
                     dim=1, index=s_idx).expand(-1, -1, c), src=src)

        return out

    # Return number of tokens not merged.
    ret_dict = {"unm_num": unm_idx.shape[1] if unm_idx.shape[1] is not None else 0}
    return merge, unmerge, ret_dict


def bipartite_soft_matching_random2d_hier(metric: torch.Tensor, frame_num: int, ratio: float, unm_pre: int, generator: torch.Generator, target_stride: int = 4, adhere_src: bool = False,  merge_mode: str = "replace", scores = None, coord = None, rec_field = 2) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape
    F = frame_num
    nf = (N - unm_pre) // F

    if ratio <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():

        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer = torch.arange(N - unm_pre, device=metric.device, dtype=torch.int64)


        # randn = torch.randint(0, F, torch.Size([nf])).to(idx_buffer) * nf
        # dst_indexes = torch.arange(nf, device=metric.device, dtype=torch.int64) + randn
        # dst_select = torch.zeros_like(idx_buffer).to(torch.bool)
        # dst_select[dst_indexes] = 1
        max_f = min(target_stride, F)
        randn = torch.randint(0, max_f, torch.Size([1]), generator=generator, device = generator.device)
        # randn = 0
        dst_select = ((torch.div(idx_buffer, nf, rounding_mode='floor')) % max_f == randn).to(torch.bool)
        # dst_select = ((idx_buffer // nf) == 0).to(torch.bool)
        a_idx = idx_buffer[None, ~dst_select, None] + unm_pre
        b_idx = idx_buffer[None, dst_select, None] + unm_pre

        unm_buffer = torch.arange(unm_pre, device=metric.device, dtype=torch.int64)[None,:,None]
        b_idx = torch.cat([b_idx, unm_buffer], dim = 1)

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices

        # We're finished with these
        del idx_buffer, unm_buffer

        num_dst = b_idx.shape[1]

        def split(x):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(x, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst
        
        def split_coord(coord):
            b, n, c = coord.shape
            src = gather(coord, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(coord, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst


        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        

        if coord is not None:
            src_coord, dst_coord = split_coord(coord)
            mask = torch.norm(src_coord[:,:,None,:] - dst_coord[:,None,:,:], dim=-1) > rec_field
            
        
        scores = a @ b.transpose(-1, -2)

        if coord is not None:
            scores[mask] = 0

        # Can't reduce more than the # tokens in src
        r = int(a.shape[1] * ratio)
        r = min(a.shape[1], r)



        if adhere_src:
            # scores = torch.sum(scores, dim=0)
            scores = torch.cat([*scores], dim = -1)
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx) % num_dst

            unm_idx = unm_idx.expand(B, -1, -1)
            src_idx = src_idx.expand(B, -1, -1)
            dst_idx = dst_idx.expand(B, -1, -1)
        else:
            # scores = torch.cat([*scores][1:], dim = -1)
            # node_max, node_idx = scores.max(dim=-1)
            # edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            # unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            # src_idx = edge_idx[..., :r, :]  # Merged Tokens
            # dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx) % num_dst

            # unm_idx = unm_idx.expand(B, -1, -1)
            # src_idx = src_idx.expand(B, -1, -1)
            # dst_idx = dst_idx.expand(B, -1, -1)


            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        # if adhere_src:
        #     unm_idx[:,...] = unm_idx[0:1]
        #     src_idx[:,...] = src_idx[0:1]
        #     dst_idx[:,...] = dst_idx[0:1]

    def merge(x: torch.Tensor, mode=None, b_select = None,  **kwarg) -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        if b_select is not None:
            if not isinstance(b_select, list):
                b_select = [b_select]
            u_idx, s_idx, d_idx = unm_idx[b_select], src_idx[b_select], dst_idx[b_select]
        else:
            u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx
        
        unm = gather(src, dim=-2, index=u_idx.expand(-1, -1, c))
        src = gather(src, dim=-2, index=s_idx.expand(-1, -1, c))
        mode = mode if mode is not None else merge_mode
        if mode != "replace":
            dst = dst.scatter_reduce(-2, d_idx.expand(-1, -1, c), src, reduce=mode, include_self=True)
        # dst = dst.scatter(-2, dst_idx.expand(n, r, c), src, reduce='add')
        
        # dst_cnt = torch.ones_like(dst)
        # src_ones = torch.ones_like(src)
        # dst_cnt = dst_cnt.scatter(-2, dst_idx.expand(n, r, c), src_ones, reduce='add')

        # dst = dst / dst_cnt
        # dst2 = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode, include_self=True)
        # assert torch.allclose(dst1, dst2)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor, b_select = None, unm_modi = None,  **kwarg) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        b, _, c = unm.shape
        if b_select is not None:
            if not isinstance(b_select, list):
                b_select = [b_select]
            u_idx, s_idx, d_idx = unm_idx[b_select], src_idx[b_select], dst_idx[b_select]
        else:
            u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx
        if unm_modi is not None:
            if unm_modi == "zero":
                unm = torch.zeros_like(unm)
        src = gather(dst, dim=-2, index=d_idx.expand(-1, -1, c))

        # Combine back to the original shape
        out = torch.zeros(b, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(b, -1, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1), dim=1, index=u_idx).expand(-1, -1, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1), dim=1, index=s_idx).expand(-1, -1, c), src=src)

        return out

    ret_dict = {"unm_num": unm_idx.shape[1]}
    return merge, unmerge, ret_dict

# For Global Token Merging.
def bipartite_soft_matching_2s( metric: torch.Tensor, 
                                src_len: int, ratio: float, align_batch: bool,
                                merge_mode: str = "replace", unmerge_chunk: int = 0) -> Tuple[Callable, Callable, dict]:
    """
    Partitions the tokens into src and dst and merges ratio of src tokens from src to dst.
    Src tokens are partitioned as first src_len tokens. Others are dst tokens.

    Args:
        - metric [B, N, C]: metric to use for similarity.
        - src_len: src token length. [ src | dst ]: [ src_len | N - src_len ]
        - ratio: ratio of src tokens to be removed (by merging).
        - unm_pre: number of src tokens not merged at previous ToMe. Pre-sequence: [unm_pre|F_0|F_1|...]
        - align_batch: whether to align similarity matching maps of samples in the batch. True when using PnP.
        - merge_mode: how to merge tokens. "mean": tokens -> Mean(src_token, dst_token); "replace": tokens -> dst_token.
        - unmerge_chunk: return which partition in unmerge. 0 for src and 1 for dst.

    Returns:
        Merge and unmerge operation according to the matching result. Return a dict including other values.
    """
    B, N, _ = metric.shape

    if ratio <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():

        idx_buffer = torch.arange(N, device=metric.device, dtype=torch.int64)

        # [ src | dst ]: [ src_len | N - src_len ]
        a_idx = idx_buffer[None, :src_len, None]
        b_idx = idx_buffer[None, src_len:, None]

        del idx_buffer

        num_dst = b_idx.shape[1]

        def split(x):
            # Split src, dst tokens
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(x, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst

        # Cosine similarity between src and dst tokens
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)

        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], int(a.shape[1] * ratio))

        if align_batch:
            # Cat scores of all samples in the batch. When using PnP, samples are (src, neg, pos).
            # Find the most similar greedily among all samples.
            scores = torch.cat([*scores], dim=-1)
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None],
                             dim=-2, index=src_idx) % num_dst # Map index to (0, num_dst - 1)
            
            # Use the same matching result for all samples
            unm_idx = unm_idx.expand(B, -1, -1)
            src_idx = src_idx.expand(B, -1, -1)
            dst_idx = dst_idx.expand(B, -1, -1)
        else:

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode=None) -> torch.Tensor:
        # Merge tokens according to matching result.
        src, dst = split(x)
        n, t1, c = src.shape
        u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx

        unm = gather(src, dim=-2, index=u_idx.expand(-1, -1, c))
        mode = mode if mode is not None else merge_mode
        if mode != "replace":
            src = gather(src, dim=-2, index=s_idx.expand(-1, -1, c))
            # In other mode such as mean, combine matched src and dst tokens.
            dst = dst.scatter_reduce(-2, d_idx.expand(-1, -1, c),
                                     src, reduce=mode, include_self=True)
        # In replace mode, just cat unmerged tokens and dst tokens. Discard src tokens.
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor, **kwarg) -> torch.Tensor:
        # Unmerge tokens to original size according to matching result.
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        b, _, c = unm.shape
        u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx
        # Restored src tokens take value from dst tokens
        src = gather(dst, dim=-2, index=d_idx.expand(-1, -1, c))

        # Combine back to the original shape
        out = torch.zeros(b, N, c, device=x.device, dtype=x.dtype)
        # Scatter dst tokens
        out.scatter_(dim=-2, index=b_idx.expand(b, -1, c), src=dst)
        # Scatter unmerged tokens
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1),
                     dim=1, index=u_idx).expand(-1, -1, c), src=unm)
        # Scatter src tokens
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1),
                     dim=1, index=s_idx).expand(-1, -1, c), src=src)
        
        out = out[:, :src_len, :] if unmerge_chunk == 0 else out[:, src_len:, :]
        return out

    ret_dict = {"unm_num": unm_idx.shape[1]}
    return merge, unmerge, ret_dict


# Original ToMe
def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(
                hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(
                sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(
            hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(
            dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(
            hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(
                h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B,
                     a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B,
                     a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge


def bipartite_soft_matching_2f(metric: torch.Tensor, src_len: int, ratio: float, adhere_src: bool, merge_mode: str = "replace", scores = None, coord = None, rec_field = 2, unmerge_chunk = 0) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if ratio <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():

        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer = torch.arange(N, device=metric.device, dtype=torch.int64)


        # randn = torch.randint(0, F, torch.Size([nf])).to(idx_buffer) * nf
        # dst_indexes = torch.arange(nf, device=metric.device, dtype=torch.int64) + randn
        # dst_select = torch.zeros_like(idx_buffer).to(torch.bool)
        # dst_select[dst_indexes] = 1
        # randn = 0
        # dst_select = ((idx_buffer // nf) == 0).to(torch.bool)
        a_idx = idx_buffer[None, :src_len, None]
        b_idx = idx_buffer[None, src_len:, None]


        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices

        # We're finished with these
        del idx_buffer

        num_dst = b_idx.shape[1]

        def split(x):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(x, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst
        
        def split_coord(coord):
            b, n, c = coord.shape
            src = gather(coord, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(coord, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst


        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        

        if coord is not None:
            src_coord, dst_coord = split_coord(coord)
            mask = torch.norm(src_coord[:,:,None,:] - dst_coord[:,None,:,:], dim=-1) > rec_field
            
        
        scores = a @ b.transpose(-1, -2)

        if coord is not None:
            scores[mask] = 0

        # Can't reduce more than the # tokens in src
        r = int(a.shape[1] * ratio)
        r = min(a.shape[1], r)



        if adhere_src:
            scores = torch.cat([*scores], dim = -1)
            # scores = torch.sum(scores, dim=0)
            node_max, node_idx = scores.max(dim=-1)

            # nscores = torch.cat([*scores], dim = -2)
            # rev_node_max, rev_node_idx = nscores.max(dim = -2)

            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx) % num_dst

            unm_idx = unm_idx.expand(B, -1, -1)
            src_idx = src_idx.expand(B, -1, -1)
            dst_idx = dst_idx.expand(B, -1, -1)
        else:
            # scores = torch.cat([*scores][1:], dim = -1)
            # node_max, node_idx = scores.max(dim=-1)
            # edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            # unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            # src_idx = edge_idx[..., :r, :]  # Merged Tokens
            # dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx) % num_dst

            # unm_idx = unm_idx.expand(B, -1, -1)
            # src_idx = src_idx.expand(B, -1, -1)
            # dst_idx = dst_idx.expand(B, -1, -1)


            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        # if adhere_src:
        #     unm_idx[:,...] = unm_idx[0:1]
        #     src_idx[:,...] = src_idx[0:1]
        #     dst_idx[:,...] = dst_idx[0:1]

    def merge(x: torch.Tensor, mode=None, b_select = None) -> torch.Tensor:

        src, dst = split(x)
        n, t1, c = src.shape
        if b_select is not None:
            if not isinstance(b_select, list):
                b_select = [b_select]
            u_idx, s_idx, d_idx = unm_idx[b_select], src_idx[b_select], dst_idx[b_select]
        else:
            u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx
        
        unm = gather(src, dim=-2, index=u_idx.expand(-1, -1, c))
        # src = gather(src, dim=-2, index=s_idx.expand(-1, -1, c))
        mode = mode if mode is not None else merge_mode
        if mode != "replace":
            dst = dst.scatter_reduce(-2, d_idx.expand(-1, -1, c), src, reduce=mode, include_self=True)
        # dst = dst.scatter(-2, dst_idx.expand(n, r, c), src, reduce='add')
        
        # dst_cnt = torch.ones_like(dst)
        # src_ones = torch.ones_like(src)
        # dst_cnt = dst_cnt.scatter(-2, dst_idx.expand(n, r, c), src_ones, reduce='add')

        # dst = dst / dst_cnt
        # dst2 = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode, include_self=True)
        # assert torch.allclose(dst1, dst2)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor, b_select = None, unm_modi = None) -> torch.Tensor:



        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        b, _, c = unm.shape
        if b_select is not None:
            if not isinstance(b_select, list):
                b_select = [b_select]
            u_idx, s_idx, d_idx = unm_idx[b_select], src_idx[b_select], dst_idx[b_select]
        else:
            u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx
        if unm_modi is not None:
            if unm_modi == "zero":
                unm = torch.zeros_like(unm)
        src = gather(dst, dim=-2, index=d_idx.expand(-1, -1, c))

        # Combine back to the original shape
        out = torch.zeros(b, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(b, -1, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1), dim=1, index=u_idx).expand(-1, -1, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1), dim=1, index=s_idx).expand(-1, -1, c), src=src)

        
        if unmerge_chunk == 0:
            out = out[:,:src_len,:]
        else:
            out = out[:,src_len:,:]

        return out

    ret_dict = {"unm_num": unm_idx.shape[1]}
    return merge, unmerge, ret_dict