# 
# Copyright (c) 2019 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the GPL License. See the LICENSE file for details.


import os

import numpy as np
import pandas as pd
import numba
import skimage

from .utils import scenes_paths, scene_id
from .img_ops import load_hr_sm



# [============================================================================]


# Baseline cPSNR values for the dataset's images. Used for normalizing scores.
# (provided by the competition's organizers)
baseline_cPSNR = pd.read_csv(
    os.path.dirname(os.path.abspath(__file__)) + '/../../Data/norm.csv',
    names = ['scene', 'cPSNR'],
    index_col = 'scene',
    sep = ' ')
    


# [============================================================================]


def score_images(imgs, scenes_paths, *args):
    """
    Measure the overall (mean) score across multiple super-resolved images.
    
    Takes as input a sequence of images (`imgs`), a sequence with the paths to
    the corresponding scenes (`scenes_paths`), and optionally a sequence of
    (hr, sm) tuples with the pre-loaded high-resolution images of those scenes.
    """
    return np.mean([
         score_image(*i)
        for i in zip(imgs, scenes_paths, *args)
        ])
    



def hr_crops(hr, sm):
    """
    "We denote the cropped 378x378 images as follows: for all u,v ∈ {0,…,6},
    HR_{u,v} is the subimage of HR with its upper left corner at coordinates
    (u,v) and its lower right corner at (378+u, 378+v)."
    -- https://kelvins.esa.int/proba-v-super-resolution/scoring/
    """
    num_cropped = 6
    max_u, max_v = np.array(hr.shape) - num_cropped
    
    for u in range(num_cropped + 1):
        for v in range(num_cropped + 1):
            yield hr[u : max_u + u, v : max_v + v], \
                  sm[u : max_u + u, v : max_v + v]
    


# [============================================================================]
def  test_score_image(sr, hr,sm,scene_id):
    """
    Calculate the individual score (cPSNR, clear Peak Signal to Noise Ratio) for
    `sr`, a super-resolved image from the scene at `scene_path`.
    
    Parameters
    ----------
    sr : matrix of shape 384x384
        super-resolved image.
    scene_path : str
        path where the scene's corresponding high-resolution image can be found.
    hr_sm : tuple, optional
        the scene's high resolution image and its status map. Loaded if `None`.
    """
    
    
    # "We assume that the pixel-intensities are represented
    # as real numbers ∈ [0,1] for any given image."

    
    # "Let N(HR) be the baseline cPSNR of image HR as found in the file norm.csv."
    N = baseline_cPSNR.loc[scene_id][0]
    
    return score_against_hr(sr, hr, sm, N)

def  score_image(sr, scene_path, hr_sm=None):
    """
    Calculate the individual score (cPSNR, clear Peak Signal to Noise Ratio) for
    `sr`, a super-resolved image from the scene at `scene_path`.
    
    Parameters
    ----------
    sr : matrix of shape 384x384
        super-resolved image.
    scene_path : str
        path where the scene's corresponding high-resolution image can be found.
    hr_sm : tuple, optional
        the scene's high resolution image and its status map. Loaded if `None`.
    """
    hr, sm = load_hr_sm(scene_path)
    
    # "We assume that the pixel-intensities are represented
    # as real numbers ∈ [0,1] for any given image."

    
    # "Let N(HR) be the baseline cPSNR of image HR as found in the file norm.csv."
    N = baseline_cPSNR.loc[scene_id(scene_path)][0]
    
    return score_against_hr(sr, hr, sm, N)
    


@numba.jit(nopython=True, parallel=True)
def score_against_hr(sr, hr, sm, N):
    """
    Numba-compiled version of the scoring function.
    """
    num_cropped = 6
    max_u, max_v = np.array(hr.shape) - num_cropped
    
    # "To compensate for pixel-shifts, the submitted images are
    # cropped by a 3 pixel border, resulting in a 378x378 format."
    c = num_cropped // 2
    sr_crop = sr[c : -c, c : -c].ravel()
    
    # create a copy of `hr` with NaNs at obscured pixels
    # (`flatten` used to bypass numba's indexing limitations)
    hr_ = hr.flatten()
    hr_[(~sm).ravel()] = np.nan
    hr = hr_.reshape(hr.shape)
    
#   crop_scores = []
    cMSEs = np.zeros((num_cropped + 1, num_cropped + 1), np.float64)
    
    for u in numba.prange(num_cropped + 1):
        for v in numba.prange(num_cropped + 1):
            
            # "We denote the cropped 378x378 images as follows: for all u,v ∈
            # {0,…,6}, HR_{u,v} is the subimage of HR with its upper left corner
            # at coordinates (u,v) and its lower right corner at (378+u, 378+v)"
            hr_crop = hr[u : max_u + u, v : max_v + v].ravel()
            
            # "we first compute the bias in brightness b"
            pixel_diff = hr_crop - sr_crop
            b = np.nanmean(pixel_diff)
            
            # "Next, we compute the corrected clear mean-square
            # error cMSE of SR w.r.t. HR_{u,v}"
            pixel_diff -= b
            pixel_diff *= pixel_diff
            cMSE = np.nanmean(pixel_diff)
            
            # "which results in a clear Peak Signal to Noise Ratio of"
#           cPSNR = -10. * np.log10(cMSE)
            
            # normalized cPSNR
#           crop_scores.append(N / cPSNR)
            
            cMSEs[u, v] = cMSE
    
    # "The individual score for image SR is"
#   sr_score = min(crop_scores)
    sr_score = N / (-10. * np.log10(cMSEs.min()))
    
    return sr_score
    


# [============================================================================]


class scorer(object):
    
    def __init__(self, scene_paths, preload_hr=True):
        """
        Wrapper to `score_image()` that simplifies the scoring of multiple
        super-resolved images.
        
        The scenes over which the scorer will operate should be given in
        `scene_paths`. This is either a sequence of paths to a subset of scenes
        or a string with a single path. In this case, it is interpreted as the
        base path to the full dataset, and `all_scenes_paths()` will be used to
        locate all the scenes it contains.
        
        Scene paths are stored in the object's `.paths` variable.
        When scoring, only the super-resolved images need to be provided.
        They are assumed to be in the same order as the scenes in `.paths`.
        
        If the object is instantiated with `preload_hr=True` (the default),
        all scene's high-resolution images and their status maps will be
        preloaded. When scoring they will be sent to `score_image()`, thus
        saving computation time in repeated scoring, at the expense of memory.
        """
        if isinstance(scene_paths, str):
            self.paths = scenes_paths(scene_paths)
        else:
            self.paths = scene_paths
        
        self.hr_sm = [] if not preload_hr else [
            highres_image(scn_path, img_as_float=True)
            for scn_path in self.paths]
        
        self.scores = []
        
    
    def __call__(self, sr_imgs, per_image=False, progbar=True, desc=''):
        """
        Score all the given super-resolved images (`sr_imgs`), which correspond
        to the scenes at the matching positions of the object's `.paths`.
        
        Returns the overall score (mean normalized cPSNR).
        
        An additional value is returned if `per_image=True`: a list with each
        image's individual cPSNR score. In either case, this list remains
        available in the object's `.scores` variable until the next call.
        """
        scenes_paths = self.paths if progbar else self.paths
        hr_sm = [] if self.hr_sm == [] else [self.hr_sm]
        
        self.scores = [
#           score_image(*i)
             score_image(*i)
            for i in zip(sr_imgs, scenes_paths, *hr_sm)]
        
        assert len(self.scores) == len(self.paths)
        
        score = np.mean(self.scores)
        
        if per_image:
            return score, self.scores
        else:
            return score
        
    