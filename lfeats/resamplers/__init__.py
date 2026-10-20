# Copyright (c) 2026 Nagoya Institute of Technology
# SPDX-License-Identifier: Apache-2.0

"""The module for audio resamplers."""

from .lilfilter import LilFilterResampler
from .manager import ResamplerManager
from .scipy import ScipyResampler
from .soxr import SoxrResampler
from .torchaudio import TorchAudioResampler

RESAMPLER_MAP = {
    "lilfilter": LilFilterResampler,
    "scipy": ScipyResampler,
    "soxr": SoxrResampler,
    "torchaudio": TorchAudioResampler,
}

__all__ = [
    "RESAMPLER_MAP",
    "ResamplerManager",
]
