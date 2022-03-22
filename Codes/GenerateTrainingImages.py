import numpy as np
from skimage.transform import resize
import tifffile
from functions import crop_images






img = tifffile.imread(r'D:\Hamed\LT2\Upper_part\lt2_upper_part_cropped_UNet.tif')
