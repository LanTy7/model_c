"""Shared utility functions used across training and evaluation scripts."""
import logging
import random

import numpy as np
import torch
import yaml


def setup_logging(log_dir: str, log_file: str = "train.log") -> logging.Logger:
    """Setup logging to file and console."""
    import os
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def safe_torch_load(path: str, map_location='cpu'):
    """Load torch checkpoint safely with weights_only=True when possible."""
    import torch
    from models.common.trainer import TrainConfig
    try:
        torch.serialization.add_safe_globals([TrainConfig])
        return torch.load(path, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError, AttributeError) as e:
        # Fallback only for serialization-related errors, not all exceptions
        import logging
        logging.getLogger(__name__).warning(
            f"weights_only=True failed ({type(e).__name__}), falling back to weights_only=False. "
            f"Only load checkpoints from trusted sources."
        )
        return torch.load(path, map_location=map_location, weights_only=False)
