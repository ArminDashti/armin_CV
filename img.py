from PIL import Image
import numpy as np
from armin_utils import os


def save_img_in_dir(array_img, saving_dir, file_name, img_format='jpg'):
    img_format = img_format.replace('.', '')
    saving_dir = saving_dir + str(file_name) + '.' + img_format
    img = Image.fromarray(array_img)
    img.save(saving_dir)
    return saving_dir
