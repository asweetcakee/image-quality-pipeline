import numpy as np
from code.logger.loggerConfig import setup_logger

logger = setup_logger(__name__)

def estimate_black_level(raw, low_percentile=5):
    f"""
        Finds a percentile point in the array
        Collects values that are lower than the percentile
        Calculates the average value of black pixels
    """

    lower_bound = np.percentile(raw, low_percentile)
    logger.info(f"Lower percentile bound: {lower_bound}")

    black_pixels = raw[raw < lower_bound]
    black_level = np.mean(black_pixels)
    logger.info(f"Black level: {black_level}")
    return black_level


def black_level_correction(raw, black_level):
    f"""
        Receives an average black_level
        Calculates black level correction into 'corrected_raw'
        Clamps out the black level to float value between 0 and 1
        Returns the corrected raw
    """
    corrected_raw = raw - black_level
    corrected_raw = np.clip(corrected_raw, 0.0, 1.0)
    return corrected_raw