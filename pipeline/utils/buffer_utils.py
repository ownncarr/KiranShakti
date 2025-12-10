# pipeline/utils/buffer_utils.py
import numpy as np
from math import cos, pi
from skimage.draw import disk


def compute_gsd_m_per_px(lat, zoom=20):
    """
    Google Maps ground sampling distance (meters per pixel)
    Formula from Google Maps API documentation.
    """
    return 156543.03392 * cos(lat * pi / 180) / (2 ** zoom)


def compute_buffer_radius_px(area_sqft, gsd_m):
    """
    Convert buffer area (sq ft) into pixel radius.
    area_sqft: buffer area in sq ft (1200 or 2400)
    gsd_m: meters per pixel
    """
    area_m2 = area_sqft * 0.092903
    pixel_area_m2 = gsd_m ** 2
    radius_px = int(np.sqrt(area_m2 / pixel_area_m2 / pi))
    return radius_px


def create_circular_buffer(img_size, radius_px):
    """
    Creates a circular binary buffer mask centered in the image.
    img_size: (H, W)
    radius_px: pixel radius
    """
    H, W = img_size
    center_y, center_x = H // 2, W // 2

    mask = np.zeros((H, W), dtype=np.uint8)
    rr, cc = disk((center_y, center_x), radius_px, shape=mask.shape)
    mask[rr, cc] = 1
    return mask


def compute_overlap_area(pv_mask_arr, buffer_mask, gsd_m):
    """
    Computes PV area inside buffer zone.
    pv_mask_arr: 0/255 mask array
    buffer_mask: 0/1 buffer array
    gsd_m: meters per pixel
    """
    pv_binary = (pv_mask_arr == 255).astype(np.uint8)
    intersection = pv_binary * buffer_mask

    pixel_area_m2 = gsd_m ** 2
    return intersection.sum() * pixel_area_m2
