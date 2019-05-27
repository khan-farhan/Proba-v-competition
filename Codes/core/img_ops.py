import os
import glob
import skimage
import numpy as np
import warnings
import scipy 

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

def process_lr_images(scene_path,processing_type = "with_same_lr",fusion_type = "median"):
    qm_images = load_lr_qm(scene_path, quality_map_only = True) 
    lr_images = load_lr_qm(scene_path, lr_only = True) 
    pxl_nc_all_lr = np.where(np.sum(qm_images,axis = 0) == 0)
    pxl_coordinates = list(zip(pxl_nc_all_lr[0],pxl_nc_all_lr[1]))

    if processing_type == "with_same_lr":
        for lr_image,qm_image in zip(lr_images,qm_images):
            lr_padded = np.pad(lr_image,((1,1),(1,1)),'constant',constant_values=(np.nan,))
            for (x,y) in pxl_coordinates:
                sub_array_xy = lr_padded[y:y+3,x:x+3]
                avg_val = np.nanmean(sub_array_xy)
                lr_image[x,y] = avg_val
                qm_image[x,y] = True

    elif processing_type == "with_all_lr":
        if fusion_type == "mean":
            lr_fused = np.nanmean(lr_images,axis = 0) 
        elif fusion_type == "mode":
            lr_fused = scipy.stats.mode(lr_image, axis=0, nan_policy='omit').mode[0]
        else:
            lr_fused = np.nanmedian(lr_images,axis = 0) 

        for lr_image,qm_image in zip(lr_images,qm_images):
            for (x,y) in pxl_coordinates:
                lr_image[x,y] = lr_fused[x,y]
                qm_image[x,y] = True

    for lr_image,qm_image in zip(lr_images,qm_images):
        lr_image[~qm_image] = np.nan

    return lr_images
 

def median_image_scene(scene_path,with_clear = False,processing_type = "with_same_lr",fusion_type = "median"):
    if not with_clear:
        lr_images = [x[0] for x in load_lr_qm(scene_path)]
        median_img_scene = np.nanmedian(lr_images,axis = 0)
        return median_img_scene
    else:
        lr_images = process_lr_images(scene_path,processing_type,fusion_type = "median")
        median_img_scene = np.nanmedian(lr_images,axis = 0)
        return median_img_scene

def mean_image_scene(scene_path,with_clear = False,processing_type = "with_same_lr",fusion_type = "mean"):
    if not with_clear:
        lr_images = [x[0] for x in load_lr_qm(scene_path)]
        mean_img_scene = np.nanmean(lr_images,axis = 0)
        return mean_img_scene
    else:
        lr_images = process_lr_images(scene_path,processing_type,fusion_type = "mean")
        mean_img_scene = np.nanmean(lr_images,axis = 0)
        return mean_img_scene


def mode_image_scene(scene_path,with_clear = False,processing_type = "with_same_lr",fusion_type = "mode"):
    if not with_clear:
        lr_images = [x[0] for x in load_lr_qm(scene_path)]
        mode_img_scene = scipy.stats.mode(lr_images, axis=0, nan_policy='omit').mode[0]
        return mode_img_scene
    else:
        lr_images = process_lr_images(scene_path,processing_type,fusion_type = "mode")
        mode_img_scene = scipy.stats.mode(lr_images, axis=0, nan_policy='omit').mode[0]
        return mode_img_scene


