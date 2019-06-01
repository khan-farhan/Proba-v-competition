import os
import glob
import skimage
import numpy as np
import warnings
import scipy 

def load_image(path,dtype):
    return skimage.io.imread(path, dtype=dtype)

def load_hr_sm(scene_path):
    """
    Loads high resolution image and it's corresponding status map given path to scene directory
    """
    hr = skimage.img_as_float64(load_image(scene_path + '/HR.png', dtype=np.uint16))
    sm = load_image(scene_path + '/SM.png', dtype=np.bool)
    return (hr, sm)


def load_lr_qm(scene_path,quality_map_only = False,lr_only = False):
    """
    Loads low resolution images and their corresponding quality map given path to scene directory
    Can also load LR images and Qm separately if the options are given
    """

    # both Lr and Qm are loaded together
    if not quality_map_only and not lr_only:
        lr_qm_images = []
        for lr_image in glob.glob(scene_path + "/LR*"):
            lr_image_path = lr_image
            qm_image_path = lr_image[:-10] + "/QM" + lr_image[-7:]
            lr = skimage.img_as_float64(load_image(lr_image_path, dtype=np.uint16)) # loading Lr image and converting it to float
            qm = load_image(qm_image_path, dtype=np.bool) # loading qm image as a boolean 
            lr_qm_images.append((lr,qm))
        return lr_qm_images

    # loading qm only
    elif quality_map_only:
        qm_images = []
        for qm_image in glob.glob(scene_path + "/QM*"):
            qm_image_path = qm_image
            qm = load_image(qm_image_path, dtype=np.bool)
            qm_images.append(qm)
        return np.asarray(qm_images)

    # loading Lr images only
    else:
        lr_images = []
        for lr_image in glob.glob(scene_path + "/LR*"):
            lr_image_path = lr_image
            lr = skimage.img_as_float64(load_image(lr_image_path, dtype=np.uint16))
            lr_images.append(lr)
        return np.asarray(lr_images)



def bicubic_upscaling(image):
    """
    Bicubic scaling of a given image by a factor of 3
    """
    return skimage.transform.rescale(image, scale=3, order=3, mode='edge',
                                  anti_aliasing=False, multichannel=False)

def upscaling_scene_images(scene_path):

    """
    Bicubic upscaling of the images. This function is provided by the competition organizers
    """
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
    """
    Processing the Low resolution images for taking care of non clear pixels.
    
    Two ways of processing which is decided by processing_type:
    1) with_same_lr: By considering a 3x3 region across the non clear pixel 
       and replacing it by the corresponding fusion_type central tendency measure(Mean, Median or Mode)

    2) with_all_lr: By considering all the lr images and taking the corresponding 
     fusion_type central tendency measure to replace the bad pixel.
    """
    qm_images = load_lr_qm(scene_path, quality_map_only = True) 
    lr_images = load_lr_qm(scene_path, lr_only = True) 

    # getting all those pixels which are bad in all the low resolution images
    pxl_nc_all_lr = np.where(np.sum(qm_images,axis = 0) == 0)
    pxl_coordinates = list(zip(pxl_nc_all_lr[0],pxl_nc_all_lr[1]))

    if processing_type == "with_same_lr":
        for lr_image,qm_image in zip(lr_images,qm_images):

            #Padding the array 
            lr_padded = np.pad(lr_image,((1,1),(1,1)),'constant',constant_values=(np.nan,))
            for (x,y) in pxl_coordinates:
                sub_array_xy = lr_padded[y:y+3,x:x+3]    # small array around the bad pixel
                avg_val = np.nanmean(sub_array_xy)       
                lr_image[x,y] = avg_val                  # Replacing the bad pixel with the mean of pixel value of nearby pixels
                qm_image[x,y] = True                     # marking it as clear pixel

    elif processing_type == "with_all_lr":
        if fusion_type == "mean":
            lr_fused = np.nanmean(lr_images,axis = 0) 
        elif fusion_type == "mode":
            lr_fused = scipy.stats.mode(lr_images, axis=0, nan_policy='omit').mode[0]
        else:
            lr_fused = np.nanmedian(lr_images,axis = 0) 

        for lr_image,qm_image in zip(lr_images,qm_images):
            for (x,y) in pxl_coordinates:
                lr_image[x,y] = lr_fused[x,y]    # replacing the bad pixel with the central tendency measure at that pixel coordinate
                qm_image[x,y] = True

    for lr_image,qm_image in zip(lr_images,qm_images):
        lr_image[~qm_image] = np.nan      # making all the remaining non clear pixels as NAN so that they won't have any impact on calculations

    return lr_images
 

def median_image_scene(scene_path,with_clear = False,processing_type = "with_same_lr",fusion_type = "median"):
    """ 
    Return a median image from all the Low resolution images of a given scene. 
    The images are also processed optionally
    """
    if not with_clear:
        lr_images = [x[0] for x in load_lr_qm(scene_path)]
        median_img_scene = np.nanmedian(lr_images,axis = 0)
        return median_img_scene
    else:
        lr_images = process_lr_images(scene_path,processing_type,fusion_type = "median")
        median_img_scene = np.nanmedian(lr_images,axis = 0)
        return median_img_scene

def mean_image_scene(scene_path,with_clear = False,processing_type = "with_same_lr",fusion_type = "mean"):

    """ 
    Return a mean image from all the Low resolution images of a given scene. 
    The images are also processed optionally
    """
    if not with_clear:
        lr_images = [x[0] for x in load_lr_qm(scene_path)]
        mean_img_scene = np.nanmean(lr_images,axis = 0)
        return mean_img_scene
    else:
        lr_images = process_lr_images(scene_path,processing_type,fusion_type = "mean")
        mean_img_scene = np.nanmean(lr_images,axis = 0)
        return mean_img_scene


def mode_image_scene(scene_path,with_clear = False,processing_type = "with_same_lr",fusion_type = "mode"):

    """ 
    Return a mode image from all the Low resolution images of a given scene. 
    The images are also processed optionally
    """
    if not with_clear:
        lr_images = [x[0] for x in load_lr_qm(scene_path)]
        mode_img_scene = scipy.stats.mode(lr_images, axis=0, nan_policy='omit').mode[0]
        return mode_img_scene
    else:
        lr_images = process_lr_images(scene_path,processing_type,fusion_type = "mode")
        mode_img_scene = scipy.stats.mode(lr_images, axis=0, nan_policy='omit').mode[0]
        return mode_img_scene


