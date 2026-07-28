"""MoMo: Disentangled Motion Modeling for Video Frame Interpolation (AAAI 2025).

Main entry point mirrors the FILM API:

    from momo import Predictor
    predictor = Predictor()
    img12 = predictor.predict(img1, img2, t)   # img1/img2: (B, H, W, 3) float32 in [0, 1]
"""

from momo.predictor import Predictor

__version__ = "0.1.0"
__all__ = ["Predictor"]
