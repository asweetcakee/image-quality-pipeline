import numpy as np
from code.logger.loggerConfig import setup_logger

logger = setup_logger(__name__)

def generate_shading_map(shape, strength=0.5):
    """
        Imitating shading map
        Uses an Euclidean distance formula to generate a shading map.
    """
    H, W = shape
    Y, X = np.ogrid[:H, :W]
    logger.info(f"Y={Y},\n X={X}")

    center_y, center_x = H / 2, W / 2
    distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    logger.info(f"distance={distance}")
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    logger.info(f"max_distance={max_distance}")

    shading_map = 1.0 - strength * (distance / max_distance)
    shading_map = np.clip(shading_map, 0.0, 1.0)

    return shading_map

def lens_shading_correction(raw, shading_map):
    """
        Corrects lens shading in a raw image
        Divides the image by the shading map to fix uneven brightness
        Values are then limited to the 0.0 to 1.0 range
    """
    corrected_raw = raw / shading_map
    corrected_raw = np.clip(corrected_raw, 0.0, 1.0)
    return corrected_raw
