"""Fast inference for SAM 2 / 2.1, applied on CUDA by SAM2ImagePredictor and SAM2VideoPredictor (inference only: the image
encoder is converted to fp16 in place). Any image / video size, nothing compiled per size, plain PyTorch ops.

Changes vs upstream README usage (bf16 autocast around everything; same network math, in fp16):
- image encoder (Hiera trunk + FPN neck) in pure fp16 (weights converted once, no per-call autocast weight casts),
  features handed on in fp32; opt-in fp16-accumulation GEMMs (not exportable to ONNX):
  `fast_encoder(model, fp16_accumulate=True)` before building the predictor
- the rest (prompt encoder, mask decoder, memory attention / encoder) under fp16 autocast (upstream: bf16 autocast)
- image preprocessing on the GPU (uint8 upload, antialiased bilinear resize to 1024 like torchvision on the CPU, normalize)
- video: JPEG frames decoded + resized with PIL exactly as upstream (bit-exact) but in a thread pool, uint8 upload,
  normalisation on the GPU
"""

import contextlib
import os

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


def _pil_frame(path, image_size):
    from PIL import Image

    with Image.open(path) as im:
        return np.array(im.convert("RGB").resize((image_size, image_size))), im.size


def load_jpeg_folder_threads(
    video_path,
    image_size,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    device="cuda",
    workers=16,
):
    """bit-exact version of sam2.utils.misc.load_video_frames_from_jpg_images (same PIL decode + resize) with the frames
    decoded in a thread pool (PIL releases the GIL), uint8 upload, /255 in float64 -> float32 and normalisation on the GPU
    """
    from concurrent.futures import ThreadPoolExecutor

    names = [
        p
        for p in os.listdir(video_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    if not names:
        raise RuntimeError(f"No JPEG frames found in {video_path}")
    names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    with ThreadPoolExecutor(min(workers, len(names))) as ex:
        res = list(
            ex.map(lambda n: _pil_frame(os.path.join(video_path, n), image_size), names)
        )
    W, H = res[0][1]
    u8 = torch.from_numpy(np.stack([r[0] for r in res])).to(device).permute(0, 3, 1, 2)
    # float32(k / 255 in float64), as upstream; GPU fp32 division is not correctly rounded
    lut = (torch.arange(256, dtype=torch.float64, device=device) / 255.0).float()
    x = torch.empty(u8.shape, dtype=torch.float32, device=device)
    for i in range(len(u8)):  # per frame: small int32 index temporaries
        x[i] = lut[u8[i].int()]
    mean = torch.tensor(img_mean, dtype=torch.float32, device=device)[:, None, None]
    std = torch.tensor(img_std, dtype=torch.float32, device=device)[:, None, None]
    x -= mean
    x /= std
    return x, H, W


def make_fast_video(pred):
    """Video predictor: fp16 encoder, fp16 autocast for the model calls, threaded frame loader. Patches pred in place."""
    if getattr(pred, "_fast_video", False) or pred.device.type != "cuda":
        return
    # init_state runs under inference_mode: convert the weights outside it (normal tensors)
    with torch.inference_mode(False):
        fast_encoder(pred)
    torch.cuda.empty_cache()  # drop the freed fp32 encoder weights (converted on the GPU)
    from .utils.misc import load_video_frames as upstream_load

    def autocast_fp16(f):
        def run(*a, **kw):
            with torch.autocast("cuda", dtype=torch.float16):
                return f(*a, **kw)

        return run

    for name in (
        "_get_image_feature",
        "_run_single_frame_inference",
        "_run_memory_encoder",
    ):
        setattr(pred, name, autocast_fp16(getattr(pred, name)))

    def load_video_frames(video_path, image_size, offload_video_to_cpu, **kw):
        if (
            isinstance(video_path, str)
            and os.path.isdir(video_path)
            and not offload_video_to_cpu
            and not kw.get("async_loading_frames")
        ):
            dev = kw.pop("compute_device", pred.device)
            kw.pop("async_loading_frames", None)
            return load_jpeg_folder_threads(video_path, image_size, device=dev, **kw)
        return upstream_load(video_path, image_size, offload_video_to_cpu, **kw)

    pred._load_video_frames = load_video_frames
    pred._fast_video = True
