from core import *
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
from subprocess import call

input_data_path = "../Data/"
output_data_path = "../Data/Median_images/"

if not os.path.exists(output_data_path):
    call( 'mkdir {}'.format(output_data_path) , shell=True )

scenes_paths = scenes_paths(input_data_path + "train")

for subpath in [input_data_path + 'train/RED/', input_data_path + 'train/NIR/']:
        for folder in os.listdir(subpath):
            median_image = median_image_scene(subpath + folder)
            img = bicubic_upscaling(median_image)
            call( 'cp {0} {1}'.format(subpath+folder+"/HR.png",output_data_path + folder + "_HR.png") , shell=True )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skimage.io.imsave(output_data_path + folder + '.png', img)
                print('*', end='', flush='True')
            
print("Finished generating images")

            




