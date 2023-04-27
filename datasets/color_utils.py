import cv2
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from einops import rearrange
import imageio
import numpy as np

import matplotlib.pyplot as plt

def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img>limit, ((img+0.055)/1.055)**2.4, img/12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img>limit, 1.055*img**(1/2.4)-0.055, 12.92*img)
    img[img>1] = 1 # "clamp" tonemapper
    return img


def read_image(img_path, img_wh, blend_a=True, exr_file = False, mapping_func = 'None'):
    if exr_file:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            img = img[..., :3]*img[..., -1:]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        '''
        NOTE: we read HDR image, but still transfer it to LDR image
        this is to reduce the impact of the huge radiance,
        which makes low quality in dark region after tone mapping.
        Though after transfering, it seems like the same as LDR image,
        but the normal LDR image is discreted with (0~255)/255.0.
        The transfered HDR image will have continued range
        '''
        if mapping_func == 'None':
            pass
        elif mapping_func == 'HDR':
            img = img / (1+img)
            img = np.power(img, 1.0/2.2)
        elif mapping_func == 'Sigmoid':
            img = 1/(1 + np.exp(-img))
        elif mapping_func == 'Tanh':
            img = np.tanh(img)
        else:
            print('Warning: mapping func not found')
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
    else:
        img = imageio.imread(img_path).astype(np.float32)/255.0
    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4: # blend A to RGB
        if blend_a:
            img = img[..., :3]*img[..., -1:]+(1-img[..., -1:])
        else:
            img = img[..., :3]*img[..., -1:]

    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')

    return img
