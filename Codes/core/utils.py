import os
import glob
import skimage
import pandas as pd
import numpy as np
import warnings
from zipfile import ZipFile
from .img_ops import *
from .SRCNN_model import *

def scenes_paths(path):
    """
    Function for returing a list of path of all the scenes directory given path to data
    """
    path_list = []
    for channel in ["/RED/","/NIR/"]:
        for scenes in os.listdir(path + channel):
            path_list.append(path +  channel + scenes)

    return path_list

def scene_id(scene_path):
    """
    Returns Id of a scene given it's scene path 
    """
    return scene_path.split('/')[-1]


def load_images_path(data_path):
    """
    Returns a list of all the LR image path given a scene directory path
    """

    images_paths = []
    for channel in ["/RED/","/NIR/"]:
        for subfolders in os.listdir(data_path + channel):
            for image_path in glob.glob(data_path + channel + subfolders + "/LR*"):
                images_paths.append(image_path)
                
    return images_paths

def load_srcnn_model(shape):
    srcnn_model = model(shape)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.2, amsgrad=False)
    srcnn_model.compile(optimizer='adam', loss="mean_squared_error", metrics=['mean_squared_error']) 
    srcnn_model.load_weights("../models/SRCNN/weights/weights.h5")
    return srcnn_model


def generate_submissions(path,out,mode):
    """
    Generate a sample submission; this function is provided on competition website
    Takes input as path to test directory and path to output directory
    """
    if not os.path.exists(out):
            os.mkdir(out)

    sub_archive = out + 'submission.zip'
    
    print('generate super resolved images: ', end='', flush='True')
    
    for subpath in [path + 'test/RED/', path + 'test/NIR/']:
        for folder in os.listdir(subpath):
            if mode == "median":
                median_image = median_image_scene(subpath + folder)
                img = bicubic_upscaling(median_image)
            elif mode == "srcnn":
                median_image = median_image_scene(subpath + folder)
                median_image = bicubic_upscaling(median_image)
                median_image = median_image.reshape(median_image.shape[0],median_image.shape[1],1)
                srcnn_model = model(median_image.shape)
                median_image = median_image.reshape(1,median_image.shape[0],median_image.shape[1],median_image.shape[2])
                img = srcnn_model.predict(median_image,batch_size = 1)[0]
                img = img.reshape(384,384)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skimage.io.imsave(out + folder + '.png', img)
            print('*', end='', flush='True')
    
    print('\narchiving: ')
    zf = ZipFile(sub_archive, mode='w')
    try:
        for img in os.listdir(out):
            # ignore the .zip-file itself
            if not img.startswith('imgset'):
                continue
            zf.write(out + '/' + img, arcname=img)
            print('*', end='', flush='True')
    finally:
        zf.close()  
    print('\ndone.')


"""
Baseline cPSNR values for the dataset's images. Used for normalizing scores.
"""
baseline_cPSNR = pd.read_csv(
    os.path.dirname(os.path.abspath(__file__)) + '/../../Data/norm.csv',
    names = ['scene', 'cPSNR'],
    index_col = 'scene',
    sep = ' ')


if __name__ == "__main__":
    print(scene_paths(path))