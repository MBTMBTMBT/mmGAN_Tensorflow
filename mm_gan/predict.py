import tensorflow as tf

from mm_gan.models import get_generator_unet
from mm_gan.data_fetcher import load_full_patient
from mm_gan.constants_and_tools import NII_SHAPE


def predict(nii_path: str, nii_shape=None):
    if nii_shape is None:
        nii_shape = NII_SHAPE
    load_full_patient(nii_paths=[nii_path], nii_shape=nii_shape)
    img_shape = (nii_shape[1], nii_shape[2])
    generator = get_generator_unet(input_shape=(4, img_shape[0], img_shape[1]), out_channels=4)
