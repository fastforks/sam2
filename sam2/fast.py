"""Fast inference for SAM 2 / 2.1, applied by SAM2ImagePredictor (inference only: the image encoder is converted to
fp16 in place). Any image size, nothing compiled per size, plain PyTorch ops.

Changes vs upstream README usage (bf16 autocast around everything; same network math, in fp16):
- image encoder (Hiera trunk + FPN neck) in pure fp16 (weights converted once, no per-call autocast weight casts),
  features handed on in fp32; opt-in fp16-accumulation GEMMs (not exportable to ONNX):
  `fast_encoder(model, fp16_accumulate=True)` before building the predictor
- the rest (prompt encoder, mask decoder) under fp16 autocast (upstream: bf16 autocast)
- image preprocessing on the GPU (uint8 upload, antialiased bilinear resize to 1024 like torchvision on the CPU, normalize)
"""

import contextlib

import numpy as np
import torch
import torch.nn.functional as F


@contextlib.contextmanager
def fp16_accumulation():
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


def fast_encoder(model, fp16_accumulate=False):
    """Make model.forward_image run the image encoder in pure fp16; outputs fp32 (as the rest of the model expects).
    Converts the model once (later calls return it unchanged)."""
    if getattr(model, "_fast_encoder", False):
        return model
    enc = model.image_encoder.half()
    dec = model.sam_mask_decoder
    pe = enc.neck.position_encoding
    pe_lp = {}

    def pe_cached(H, W, device):
        if (H, W) not in pe.cache:
            pe._pe(1, device, H, W)
        return pe.cache[(H, W)].to(device)  # fp32 [C, H, W]

    def pe_forward(x):
        """position encodings without the per-call repeat + dtype copies (input-independent, cached per size / dtype)"""
        H, W = x.shape[-2:]
        k = (H, W, x.dtype, x.device)
        if k not in pe_lp:
            pe_lp[k] = pe_cached(H, W, x.device).to(x.dtype)
        return pe_lp[k][None].expand(x.shape[0], -1, -1, -1)

    pe.forward = pe_forward

    def forward_image(img_batch):
        acc = fp16_accumulation() if fp16_accumulate else contextlib.nullcontext()
        with torch.autocast("cuda", enabled=False), acc:
            out = enc(img_batch.half())
        out = dict(out)
        # high-res levels stay fp16 (consumed by conv_s0 / conv_s1 under fp16 autocast: no fp32 round trip of the
        # 256x256x256 map); the lowest-res level and the position encodings go on in fp32 as upstream expects
        out["backbone_fpn"] = [
            f if i < 2 and model.use_high_res_features_in_sam else f.float()
            for i, f in enumerate(out["backbone_fpn"])
        ]
        out["vision_pos_enc"] = [
            pe_cached(p.shape[-2], p.shape[-1], p.device)[None].expand(
                p.shape[0], -1, -1, -1
            )
            for p in out["vision_pos_enc"]
        ]  # fp32 cache views (= upstream values)
        out["vision_features"] = out["backbone_fpn"][-1]
        if model.use_high_res_features_in_sam:
            with torch.autocast("cuda", dtype=torch.float16):
                out["backbone_fpn"][0] = dec.conv_s0(out["backbone_fpn"][0])
                out["backbone_fpn"][1] = dec.conv_s1(out["backbone_fpn"][1])
        return out

    model.forward_image = forward_image
    model._fast_encoder = True
    return model


def preprocess(image, resolution, mean, std, device):
    """uint8 HxWx3 numpy / PIL image -> normalised fp32 [1, 3, R, R] on the GPU (mean / std scaled to [0, 255])"""
    image = np.ascontiguousarray(image)
    if not image.flags.writeable:  # PIL images give read-only arrays
        image = image.copy()
    x = torch.from_numpy(image).to(device).permute(2, 0, 1)[None]
    x = F.interpolate(
        x.float(),
        (resolution, resolution),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (x - mean) / std


def to_host(t):
    """float32 numpy copy of a GPU tensor via a pinned staging tensor (cached by the pinned host allocator)"""
    t = t.detach().float()
    if t.device.type != "cuda":
        return t.cpu().numpy()
    h = torch.empty(t.shape, dtype=torch.float32, pin_memory=True)
    h.copy_(t)
    return h.numpy()
