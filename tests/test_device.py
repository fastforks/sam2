"""CPU fallback and per-instance video loader; no GPU or checkpoint needed."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sam2 import fast
from sam2.sam2_image_predictor import SAM2ImagePredictor


class CPUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 32
        self.image_encoder = torch.nn.Conv2d(3, 4, 1)

    @property
    def device(self):
        return next(self.parameters()).device


def test_cpu_predictor_preserves_precision_and_preprocessing():
    model = CPUModel()
    predictor = SAM2ImagePredictor(model)
    assert next(model.image_encoder.parameters()).dtype == torch.float32
    image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    expected = predictor._transforms(image)[None]
    torch.testing.assert_close(predictor._preprocess(image), expected, rtol=0, atol=0)
    tensor = torch.arange(6, dtype=torch.float32, requires_grad=True)
    np.testing.assert_array_equal(fast.to_host(tensor), np.arange(6, dtype=np.float32))


def test_threaded_loader_matches_upstream_on_cpu(tmp_path):
    from sam2.utils.misc import load_video_frames_from_jpg_images

    rng = np.random.default_rng(42)
    for i in range(3):
        Image.fromarray(rng.integers(0, 256, (18, 24, 3), dtype=np.uint8)).save(
            tmp_path / f"{i:05d}.jpg"
        )
    expected, h, w = load_video_frames_from_jpg_images(
        str(tmp_path), 32, offload_video_to_cpu=True
    )
    actual, ah, aw = fast.load_jpeg_folder_threads(str(tmp_path), 32, device="cpu")
    assert (ah, aw) == (h, w)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_empty_frame_folder_has_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="No JPEG frames"):
        fast.load_jpeg_folder_threads(str(tmp_path), 32, device="cpu")


def test_video_loaders_are_instance_local_and_prepare_is_idempotent(
    monkeypatch, tmp_path
):
    from sam2 import sam2_video_predictor as video_module
    from sam2.utils import misc

    upstream = video_module.load_video_frames
    monkeypatch.setattr(fast, "fast_encoder", lambda model: model)
    monkeypatch.setattr(
        fast, "load_jpeg_folder_threads", lambda *a, **kw: ("threads", kw["device"])
    )
    monkeypatch.setattr(misc, "load_video_frames", lambda *a, **kw: ("upstream", kw))

    def model(device):
        return SimpleNamespace(
            device=torch.device(device),
            _get_image_feature=lambda: None,
            _run_single_frame_inference=lambda: None,
            _run_memory_encoder=lambda: None,
        )

    first, second = model("cuda:0"), model("cuda:1")
    fast.make_fast_video(first)
    wrapped = first._get_image_feature
    fast.make_fast_video(first)
    assert first._get_image_feature is wrapped
    fast.make_fast_video(second)
    assert video_module.load_video_frames is upstream
    assert first._load_video_frames(str(tmp_path), 32, False) == (
        "threads",
        torch.device("cuda:0"),
    )
    assert second._load_video_frames(str(tmp_path), 32, False) == (
        "threads",
        torch.device("cuda:1"),
    )
    assert (
        first._load_video_frames(str(tmp_path), 32, False, async_loading_frames=True)[0]
        == "upstream"
    )
    assert first._load_video_frames(str(tmp_path), 32, True)[0] == "upstream"


def test_cpu_video_builder_and_init_keep_upstream_precision(tmp_path, monkeypatch):
    from sam2.build_sam import build_sam2_video_predictor

    model = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml", device="cpu"
    )
    assert next(model.image_encoder.parameters()).dtype == torch.float32
    monkeypatch.setattr(model, "_get_image_feature", lambda *a, **kw: None)
    Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)).save(tmp_path / "00000.jpg")
    state = model.init_state(str(tmp_path))
    assert state["device"].type == "cpu"
    assert state["images"].device.type == "cpu"
    assert not getattr(model, "_fast_video", False)
