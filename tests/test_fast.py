"""Tests for the fast path of SAM2ImagePredictor / SAM2VideoPredictor (sam2/fast.py): parity with the upstream
flow (README usage, bf16 autocast).

Needs a CUDA GPU and a SAM 2.1 checkpoint: SAM2_CHECKPOINT=/path/to/sam2.1_hiera_tiny.pt (config from SAM2_CONFIG,
default configs/sam2.1/sam2.1_hiera_t.yaml). The loader test needs no checkpoint.
"""

import contextlib
import os
import shutil

import numpy as np
import pytest
import torch
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get("SAM2_CHECKPOINT", "")
CFG = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
ckpt = pytest.mark.skipif(not os.path.isfile(CKPT), reason="set SAM2_CHECKPOINT")


def _iou(a, b):
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    return (a & b).sum() / max((a | b).sum(), 1)


@pytest.fixture
def frames(tmp_path):
    src = os.path.join(ROOT, "notebooks", "videos", "bedroom")
    for i in range(8):
        shutil.copy(os.path.join(src, f"{i:05d}.jpg"), tmp_path)
    return str(tmp_path)


@cuda
def test_threaded_loader_is_bit_exact(frames):
    from sam2.fast import load_jpeg_folder_threads
    from sam2.utils.misc import load_video_frames_from_jpg_images

    ref, h, w = load_video_frames_from_jpg_images(
        frames, 1024, offload_video_to_cpu=True
    )
    out, h2, w2 = load_jpeg_folder_threads(frames, 1024)
    assert (h, w) == (h2, w2)
    assert torch.equal(out.cpu(), ref)


def _upstream_predict(model, image, box):
    """the upstream SAM2ImagePredictor set_image + predict(box) flow under bf16 autocast (README usage)"""
    from sam2.utils.transforms import SAM2Transforms

    tf = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0)
    hw = image.shape[:2]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        backbone_out = model.forward_image(tf(image)[None].cuda())
        _, feats, _, sizes = model._prepare_backbone_features(backbone_out)
        feats[-1] = feats[-1] + model.no_mem_embed
        feats = [f.permute(1, 2, 0).view(1, -1, *s) for f, s in zip(feats, sizes)]
        corners = tf.transform_boxes(
            torch.as_tensor(box, dtype=torch.float, device="cuda"),
            normalize=True,
            orig_hw=hw,
        ).reshape(-1, 2, 2)
        labels = torch.tensor([[2, 3]], dtype=torch.int, device="cuda")
        sparse, dense = model.sam_prompt_encoder(
            points=(corners, labels), boxes=None, masks=None
        )
        low, iou, _, _ = model.sam_mask_decoder(
            image_embeddings=feats[-1],
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=feats[:-1],
        )
        masks = tf.postprocess_masks(low, hw) > 0
    return masks[0].float().cpu().numpy(), iou[0].float().cpu().numpy()


@cuda
@ckpt
@pytest.mark.parametrize("fp16_accumulate", [False, True])
def test_image_matches_upstream(fp16_accumulate):
    from sam2.build_sam import build_sam2
    from sam2.fast import fast_encoder
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image = np.array(
        Image.open(os.path.join(ROOT, "notebooks", "images", "truck.jpg")).convert(
            "RGB"
        )
    )
    box = np.array([425, 600, 700, 875])
    model = build_sam2(CFG, CKPT)
    m_ref, s_ref = _upstream_predict(model, image, box)
    fast = SAM2ImagePredictor(fast_encoder(model, fp16_accumulate=fp16_accumulate))
    fast.set_image(image)
    m, s, low = fast.predict(box=box, multimask_output=False)
    fast.set_image(Image.fromarray(image))  # PIL input, as upstream
    assert np.array_equal(fast.predict(box=box, multimask_output=False)[0], m)
    assert m.shape == m_ref.shape and m.dtype == m_ref.dtype and low.dtype == np.float32
    assert _iou(m[0], m_ref[0]) > 0.98
    assert abs(float(s[0]) - float(s_ref[0])) < 0.02
    fast.set_image_batch([image, image[::2, ::2].copy()])
    mb, sb, _ = fast.predict_batch(box_batch=[box, box // 2], multimask_output=False)
    assert _iou(mb[0][0], m[0]) > 0.99  # batched GEMMs: not bit-identical
    assert mb[1].shape[-2:] == (image.shape[0] // 2, image.shape[1] // 2)


class UpstreamVideoPredictor(SAM2VideoPredictor):
    """subclasses keep the upstream path (see SAM2VideoPredictor.__init__)"""


@cuda
@ckpt
def test_video_matches_upstream(frames):
    from sam2.build_sam import build_sam2_video_predictor

    def track(pred, ctx):
        out = {}
        with ctx:
            state = pred.init_state(frames)
            pred.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=1,
                points=np.array([[210, 350]], np.float32),
                labels=np.array([1], np.int32),
            )
            for f, _, logits in pred.propagate_in_video(state):
                assert logits.dtype == torch.float32
                out[f] = (logits[0, 0] > 0).cpu().numpy()
        return out

    up = f"++model._target_={__name__}.UpstreamVideoPredictor"
    ref = track(
        build_sam2_video_predictor(CFG, CKPT, hydra_overrides_extra=[up]),
        torch.autocast("cuda", dtype=torch.bfloat16),
    )
    fast = track(build_sam2_video_predictor(CFG, CKPT), contextlib.nullcontext())
    assert sorted(fast) == sorted(ref)
    ious = [_iou(fast[f], ref[f]) for f in ref]
    # small object: single frames move by a few boundary pixels (upstream bf16 vs fp32 differs as much)
    assert np.mean(ious) > 0.98 and min(ious) > 0.9, ious
