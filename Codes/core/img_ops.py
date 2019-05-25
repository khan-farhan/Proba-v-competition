import os
import glob
import skimage
import numpy as np
import warnings

def load_image(path,dtype):
    return skimage.io.imread(path, dtype=dtype)

def load_hr_sm(scene_path):
    hr = skimage.img_as_float64 (load_image(scene_path + '/HR.png', dtype=np.uint16))
    sm = load_image(scene_path + '/SM.png', dtype=np.bool)
    return (hr, sm)

def load_lr_qm(scene_path,quality_map_only = False,lr_only = False):
    if not quality_map_only and not lr_only:
        lr_qm_images = []
        for lr_image in glob.glob(scene_path + "/LR*"):
            lr_image_path = lr_image
            qm_image_path = lr_image[:-10] + "/QM" + lr_image[-7:]
            lr = skimage.img_as_float64(load_image(lr_image_path, dtype=np.uint16))
            qm = load_image(qm_image_path, dtype=np.bool)
            lr_qm_images.append((lr,qm))
        return lr_qm_images
    elif quality_map_only:
        qm_images = []
        for qm_image in glob.glob(scene_path + "/QM*"):
            qm_image_path = qm_image
            qm = load_image(qm_image_path, dtype=np.bool)
            qm_images.append(qm)
        return np.asarray(qm_images)
    else:
        lr_images = []
        for lr_image in glob.glob(scene_path + "/LR*"):
            lr_image_path = lr_image
            lr = skimage.img_as_float64(load_image(lr_image_path, dtype=np.uint16))
            lr_images.append(lr)
        return np.asarray(lr_images)



def bicubic_upscaling(image):
    return skimage.transform.rescale(image, scale=3, order=3, mode='edge',
                                  anti_aliasing=False, multichannel=False)

def upscaling_scene_images(scene_path):
    clearance = []
    for lrc_fn in glob.glob(scene_path + '/QM*.png'):
        lrc = load_image(lrc_fn, dtype=np.bool)
        clearance.append( (np.sum(lrc), lrc_fn[-7:-4]) )

            # determine subset of images of maximum clearance 
        maxcl = max([x[0] for x in clearance])
        maxclears = [x[1] for x in clearance if x[0] == maxcl]

            # upscale and aggregate images with maximum clearance together
        img = np.zeros( (384, 384), dtype=np.float)
        for idx in maxclears:
            lrfn = 'LR{}.png'.format(idx)

            lr = load_image('/'.join([scene_path, lrfn]), dtype=np.uint16)
            lr_float = skimage.img_as_float(lr)

                # bicubic upscaling
            img += skimage.transform.rescale(lr_float, scale=3, order=3, mode='edge', anti_aliasing=False, multichannel=False)
        img /= len(maxclears)
    return img

def median_scene(scene_path):
    lr_images = [x[0] for x in load_lr_qm(scene_path)]
    median_img_scene = np.nanmedian(lr_images,axis = 0)
    return median_img_scene