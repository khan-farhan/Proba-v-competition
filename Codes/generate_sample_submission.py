import warnings
import os, sys
from glob import glob 

# See https://scikit-image.org
import skimage
from skimage import io 
from skimage.transform import rescale

import numpy as np 
from zipfile import ZipFile


def main(path, out):
    # name of submission archive
    sub_archive = out + '/submission.zip'
    
    print('generate sample solutions: ', end='', flush='True')
    
    for subpath in [path + '/test/RED', path + '/test/NIR']:
        for folder in os.listdir(subpath):
            clearance = []
            for lrc_fn in glob(subpath + '/' + folder + '/QM*.png'):
                lrc = io.imread(lrc_fn, dtype=np.bool)
                clearance.append( (np.sum(lrc), lrc_fn[-7:-4]) )

            # determine subset of images of maximum clearance 
            maxcl = max([x[0] for x in clearance])
            maxclears = [x[1] for x in clearance if x[0] == maxcl]

            # upscale and aggregate images with maximum clearance together
            img = np.zeros( (384, 384), dtype=np.float)
            for idx in maxclears:
                lrfn = 'LR{}.png'.format(idx)

                lr = io.imread('/'.join([subpath, folder, lrfn]), dtype=np.uint16)
                lr_float = skimage.img_as_float(lr)

                # bicubic upscaling
                img += rescale(lr_float, scale=3, order=3, mode='edge', anti_aliasing=False, multichannel=False)
            img /= len(maxclears)
            
            # normalize and safe resulting image in temporary folder (complains on low contrast if not suppressed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(out + '/' + folder + '.png', img)
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
        
        
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: python {0} [path to testfolder] [name of the submission folder]\n".format(sys.argv[0]))
        print("EXAMPLE: python {0} data submission\n".format(sys.argv[0]))
    else:
        _, path, out = sys.argv
        
        # sanity check
        if 'test' not in os.listdir(path):
            raise ValueError('ABORT: your path {} does not contain a folder "test".'.format(path))
        
        # creating folder for convenience
        if out not in os.listdir('.'):
            os.mkdir(out)
        
        main(path, out)