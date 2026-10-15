# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the ReDimNet2+ model."""

from enum import Enum

import torch
from huggingface_hub import hf_hub_download

from ..interfaces.types import Audio, Features
from ..utils.io import safe_torch_hub_load, silence_hf_hub
from ..utils.validation import validate_enum, validate_length
from .base import UtteranceLevelFeatureModel


class ReDimNet2PlusVariant(str, Enum):
    """Enumeration of supported ReDimNet2+ model variants."""

    B6 = "b6"


class ReDimNet2PlusModel(UtteranceLevelFeatureModel):
    """A class for the ReDimNet2+ model.

    The architecture is identical to that of the public ReDimNet2, and only the
    parameters are replaced with the fine-tuned ones.

    """

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the ReDimNet2+ model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(
            variant, ReDimNet2PlusVariant, ReDimNet2PlusVariant.B6
        )
        self._model_id = f"redimnet2+-{self.variant.value}"

    def load(self, model_dir: str, quiet: bool = False) -> None:
        """Load the model from the specified directory.

        Parameters
        ----------
        model_dir : str
            The directory where the model checkpoint will be stored.

        quiet : bool, optional
            Whether to suppress output during the loading process.

        """
        if self.model is not None:
            return

        # Build the model from the public ReDimNet2 with the same architecture.
        model = safe_torch_hub_load(
            "PalabraAI/redimnet2:main",
            "redimnet2",
            model_dir,
            quiet=quiet,
            model_name=self.variant.value,
            train_type="lm",
            dataset="vox2",
        )

        with silence_hf_hub(quiet):
            model_path = hf_hub_download(
                repo_id="lab260/redimnet2-plus",
                filename="pytorch_model_fsdp.bin",
                repo_type="model",
                cache_dir=model_dir,
            )

        # The checkpoint contains the whole training model including the classifier.
        checkpoint = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=True
        )
        prefix = "_orig_mod.encoder.model."
        state_dict = {
            k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)
        }
        model.load_state_dict(state_dict, strict=True)

        self.model = model
        self.model.eval()
        self.model.to(self.device)

    def extract_features_impl(self, audio: Audio, layers: list[int]) -> Features:
        """Extract features from the input audio using the model.

        Parameters
        ----------
        audio : Audio
            The input audio data with shape (B, T).

        layers : list[int]
            The layer(s) from which to extract features.

        Returns
        -------
        out : Features
            The extracted features.

        Raises
        ------
        RuntimeError
            If the model is not loaded.

        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call 'load' method first.")

        with torch.inference_mode():
            inputs = audio.tensor.to(self.device)
            inputs = validate_length(inputs, 720)
            vectors = self.model(inputs)
            vectors = vectors.unsqueeze(1)

        return Features(data=vectors, source=self.model_id)
