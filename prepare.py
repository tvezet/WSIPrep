from histolab.slide import Slide
from histolab.masks import TissueMask
from PIL import Image
import numpy as np
import glob
import os.path as path
import cv2

TBN_SCALE = 64

SCALE_FACTOR = 32
TISSUE_ONLY = True
SEPARATE = True
SAVE_THUMB = True
SOURCE_EXT = '.tif'
TARGET_EXT = '.png'

R_PATH = path.join('data','raw')
P_PATH = path.join('data','processed')

def print_mask(slide,mask):
    overview = slide.scaled_image(TBN_SCALE)
    overview.putalpha(Image.new('L', overview.size, 255))
    backgr = Image.new('RGB', (mask.shape[1],mask.shape[0]), (255,0,0))
    backgr.putalpha(Image.fromarray(np.where(mask,0,128).astype(np.uint8)))
    backgr = backgr.resize(overview.size)
    return Image.alpha_composite(overview,backgr)
    
def mask_crop(image,mask):
    # calculate tissue boundaries
    mheight, mwidth = mask.shape
    xidces = np.ones_like(mask)*np.arange(mwidth)[np.newaxis,:]
    left = np.min(xidces[mask])
    right = np.max(xidces[mask])
    yidces = np.ones_like(mask)*np.arange(mheight)[:,np.newaxis]
    bot = np.min(yidces[mask])
    top = np.max(yidces[mask])
    # translate boundaries to image size
    width, height = image.size
    xscale = width/mwidth
    yscale = height/mheight
    return image.crop((left*xscale,bot*yscale,right*xscale,top*yscale))
    
def tissue_crops(slide):
    # create mask
    masker = TissueMask()
    mask = masker(slide)
    # scale image
    scaled = slide.scaled_image(SCALE_FACTOR)
    if SAVE_THUMB:
        thumb = print_mask(slide,mask)
    else:
        thumb = None
    if SEPARATE:
        # divide in different components
        nLabels, segments, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
        crops = []
        for i in range(1,nLabels):
            mask = segments == i
            crops.append(mask_crop(scaled,mask))
    else:
        crops = [mask_crop(scaled,mask)]
    return thumb, crops

for fname in glob.glob(path.join(R_PATH,'*'+SOURCE_EXT)):
    no_dir_no_ext = '.'.join(path.split(fname)[1].split('.')[:-1])
    try:
        slide = Slide(fname, processed_path=P_PATH)
        if TISSUE_ONLY:
            thumb, crops = tissue_crops(slide)
            if thumb is not None:
                thumb.save(path.join(P_PATH, no_dir_no_ext+ '_thumb'+TARGET_EXT))
            if len(crops)==1:
                crops[0].save(path.join(P_PATH, no_dir_no_ext+TARGET_EXT))
            else:
                for i, crop in enumerate(crops):
                    crop.save(path.join(P_PATH, no_dir_no_ext+'_'+str(i)+TARGET_EXT))
        else:
            scaled = slide.scaled_image(SCALE_FACTOR)
            scaled.save(path.join(P_PATH, no_dir_no_ext+TARGET_EXT))
        print('File "' + fname + '" done.')
    except Exception as e:
        print('Could not handle file "' + fname + '"')
        raise e




