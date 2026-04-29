import cv2
import numpy as np
from skimage.exposure import match_histograms

def apply_gamma(img, gamma):
    if gamma == 1.0:
        return img
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255
    return cv2.LUT(img, table.astype(np.uint8))

def apply_contrast(img, alpha):
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

def apply_sharpen(img, strength):
    if strength == 0:
        return img
    blur = cv2.GaussianBlur(img, (0,0), 3)
    return cv2.addWeighted(img, 1 + strength, blur, -strength, 0)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def apply_histogram_matching(img, ref, strength=1.0):
    matched = match_histograms(img, ref, channel_axis=-1).astype(np.uint8)
    return (img * (1-strength) + matched * strength).astype(np.uint8)
