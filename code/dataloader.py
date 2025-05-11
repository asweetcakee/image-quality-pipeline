import imageio
import numpy as np
from pathlib import Path

from code.logger.loggerConfig import setup_logger

logger = setup_logger(__name__)

class SimpleRawDataset:
    def __init__(self, root, split="train"):
        base = Path(root) / split
        self.raw_paths = sorted((base / "raw").glob("*.png"))
        self.gt_paths = sorted((base / "gt").glob("*.jpg"))
        logger.info(f"Found {self.raw_paths} | len(raw_paths) = {len(self.raw_paths)} paths")
        logger.info(f"Found {self.gt_paths} | len(gt_paths) = {len(self.gt_paths)} paths")

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, idx):
        # Loads raw Bayer-patterned PNG
        raw = self.__normalize(self.raw_paths[idx])

        # Loads ground-truth RGB
        gt = self.__normalize(self.gt_paths[idx])

        logger.info(f"Raw shape: {raw.shape}, dtype: {raw.dtype}")
        logger.info(f"GT shape: {gt.shape}, dtype: {gt.dtype}")
        return raw, gt

    def __normalize(self, path):
        data_matrix = imageio.imread(path)

        # Normalizes to [0,1]
        max_val = np.iinfo(data_matrix.dtype).max
        data_matrix = data_matrix.astype(np.float32) / max_val

        return data_matrix