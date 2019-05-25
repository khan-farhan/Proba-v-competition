import os
import glob
import skimage
import pandas as pd
import numpy as np
import warnings
from zipfile import ZipFile

def scenes_paths(path):
    path_list = []
    for channel in ["/RED/","/NIR/"]:
        for scenes in os.listdir(path + channel):
            path_list.append(path +  channel + scenes)

    return path_list

def scene_id(scene_path):
    return scene_path.split('/')[-1]


def generate_submissions(path,out):

    sub_archive = out + 'submission.zip'
    
    print('generate sample solutions: ', end='', flush='True')
    
    for subpath in [path + 'test/RED/', path + 'test/NIR/']:
        for folder in os.listdir(subpath):
            median_image = median_scene(subpath + folder)
            img = bicubic_upscaling(median_image)
        
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
    print('\ndone. The submission-file is found at {}. Bye!'.format(sub_archive))

def load_images_path(data_path):
    images_paths = []
    for channel in ["/RED/","/NIR/"]:
        for subfolders in os.listdir(data_path + channel):
            for image_path in glob.glob(data_path + channel + subfolders + "/LR*"):
                images_paths.append(image_path)
                
    return images_paths

baseline_cPSNR = pd.read_csv(
    os.path.dirname(os.path.abspath(__file__)) + '/../../Data/norm.csv',
    names = ['scene', 'cPSNR'],
    index_col = 'scene',
    sep = ' ')


if __name__ == "__main__":
    print(scene_paths(path))