import matplotlib.pyplot as plt
from code.dataloader import SimpleRawDataset
from code.pipeline.blackLevel import black_level_correction, estimate_black_level
from code.pipeline.lensShading import generate_shading_map, lens_shading_correction
from code.logger.loggerConfig import setup_logger

logger = setup_logger(__name__)

def main():
    root = "data"

    # Initializes the data set
    ds = SimpleRawDataset(root=root, split="train")
    raw, gt = ds[0] # loads first RAW and GT images
    logger.info(f"raw: {raw}")

    # Black Level Correction
    black_lvl = estimate_black_level(raw)
    raw_blc = black_level_correction(raw, black_lvl)
    logger.info(f"Raw - black level corrected: \n {raw_blc}")

    # Lens Shading Correction
    shading_map = generate_shading_map(raw_blc.shape)
    logger.info(f"Shading map: {shading_map}")

    raw_lsc = lens_shading_correction(raw_blc, shading_map)
    logger.info(f"Raw - lens shading corrected: \n {raw_lsc}")


if __name__ == "__main__":
    main()