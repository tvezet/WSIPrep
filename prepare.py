from histolab.slide import Slide
from histolab.masks import TissueMask
from PIL import Image
import numpy as np
import glob
import os.path as path

SCALE_FACTOR = 32
TISSUE_ONLY = True
TARGET_EXT = '.png'

R_PATH = path.join('data','raw')
P_PATH = path.join('data','processed')

def get_tissue_crop(slide):
    masker = TissueMask()
    scaled = slide.scaled_image(SCALE_FACTOR)
    mask = masker(slide)
    mheight, mwidth = mask.shape
    xidces = np.ones_like(mask)*np.arange(mwidth)[np.newaxis,:]
    left = np.min(xidces[mask])
    right = np.max(xidces[mask])
    yidces = np.ones_like(mask)*np.arange(mheight)[:,np.newaxis]
    bot = np.min(yidces[mask])
    top = np.max(yidces[mask])
    width, height = scaled.size
    xscale = width/mwidth
    yscale = height/mheight
    return scaled.crop((left*xscale,bot*yscale,right*xscale,top*yscale))

for fname in glob.glob(path.join(R_PATH,'*')):
    try:
        slide = Slide(fname, processed_path=P_PATH)
        if TISSUE_ONLY:
            scaled = get_tissue_crop(slide)
        else:
            scaled = slide.scaled_image(SCALE_FACTOR)
        no_dir_no_ext = '.'.join(path.split(fname)[1].split('.')[:-1])
        scaled.save(path.join(P_PATH, no_dir_no_ext+TARGET_EXT))
    except Exception as e:
        print('Could not handle file "' + fname + '"')




