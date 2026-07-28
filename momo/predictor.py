"""google_film-compatible wrapper around the MoMo frame interpolator.

Usage mirrors `from google_film import Predictor`:

    from momo import Predictor
    predictor = Predictor()  # defaults to the full MoMo checkpoint on /fsx_scanline
    img12 = predictor.predict(img1, img2, t)   # img1/img2: (B, H, W, 3) float32 in [0, 1]

Note: MoMo is a single-midpoint interpolator, so only ``t == 0.5`` is
supported. Values that deviate from 0.5 raise a warning and are treated as the
midpoint.
"""
import warnings

import numpy as np
import torch

from momo.synthesis import SynthesisNet
from momo.diffusion.momo import MoMo


class Predictor:
    def __init__(self,
                 checkpoint_path: str = "/fsx_scanline/from_eyeline/users/lima/pretrained/MoMo_weights/experiments/diffusion/momo_full/weights/model.pth",
                 fp16: bool = False,
                 gpu_idx: int = 0,
                 num_inference_steps: int = 8,
                 resize_to_fit: bool = True,
                 pad_to_fit_unet: bool = False):
        if torch.cuda.is_available():
            if gpu_idx >= torch.cuda.device_count():
                raise ValueError(
                    f"gpu_idx {gpu_idx} is out of range. "
                    f"Available GPUs: {torch.cuda.device_count()}")
            self.device = torch.device(f"cuda:{gpu_idx}")
        else:
            self.device = torch.device("cpu")

        self.fp16 = fp16
        self.num_inference_steps = num_inference_steps
        self.resize_to_fit = resize_to_fit
        self.pad_to_fit_unet = pad_to_fit_unet

        # build the full MoMo model (synthesis net + diffusion flow model)
        synth_model = SynthesisNet()
        self.model = MoMo(synth_model=synth_model)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        del ckpt

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self,
                srgb1: np.ndarray,  # b, h, w, 3
                srgb2: np.ndarray,  # b, h, w, 3
                t: np.ndarray,      # b,
                ) -> np.ndarray:    # b, h, w, 3
        t = np.asarray(t, dtype=np.float32).reshape(-1)
        if not np.allclose(t, 0.5):
            warnings.warn(
                "MoMo only supports midpoint interpolation (t=0.5); "
                "the provided `t` is ignored and treated as 0.5.",
                stacklevel=2)

        # (b, h, w, 3) -> (b, 3, h, w)
        frame0 = torch.from_numpy(np.ascontiguousarray(srgb1)).float().permute(0, 3, 1, 2).to(self.device)
        frame1 = torch.from_numpy(np.ascontiguousarray(srgb2)).float().permute(0, 3, 1, 2).to(self.device)

        # MoMo expects the two frames stacked along a frame dim: (b, 3, 2, h, w)
        x = torch.stack([frame0, frame1], dim=2)

        with torch.autocast(device_type=self.device.type, enabled=self.fp16):
            mid_frame, _ = self.model(
                x,
                num_inference_steps=self.num_inference_steps,
                resize_to_fit=self.resize_to_fit,
                pad_to_fit_unet=self.pad_to_fit_unet,
            )

        # (b, 3, h, w) -> (b, h, w, 3), clamp to valid image range
        mid_frame = mid_frame.clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy()
        return mid_frame
