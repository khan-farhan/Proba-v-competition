This repository contains jupyter notebooks and python scripts for analysis of problem described in competition [PROBA-V Super Resolution](https://kelvins.esa.int/proba-v-super-resolution/home/). 

#### Directory structure:
```sh
tree -L 2
├── Codes
│   ├── EDA_V1.ipynb
│   ├── EDA_V2.ipynb
│   ├── EDA_V3.ipynb
│   ├── SRCNN_test.ipynb
│   ├── SRCNN_train.ipynb
│   ├── Statistical_experimentation.ipynb
│   ├── core
│   ├── generate_median_images.py
│   └── generate_sample_submission.py
├── Data
│   ├── Median_images
│   ├── norm.csv
│   ├── test
│   └── train
├── Result
│   ├── Median_images
│   ├── SRCNN_images
│   ├── mean_sq_err.png
│   └── val_loss.png
└── models
    └── SRCNN
```

#### Description of each script is as follows:
- **EDA_V1.ipynb** : Exploratory data analysis of Data directory
- **EDA_V1.ipynb**: Visualizing given low resolution and high resolution images
- **EDA_V3.ipynb** : Analyzing the quality of low resolution images(bad pixels, NAN values present etc.) 
- **Statistical_experimentation.ipynb**: Generating super resolution images using central tendency measures.
- **SRCNN_train.ipynb** : Training on SRCNN.
- **SRCNN_test.ipynb** : Testing the performance of trained SRCNN network by calculating cPSNR score.
- **generate_median_images.py** : Script for generating images for training SRCNN.
- **generate_sample_submission.py** : Utility script for generating submission results on competition website.

#### Training environment:
Training is done on AWS ec2 instance with Deep learning ami already configured with TensorFlow and Keras 2 on Python 3 with CUDA 10.0 and MKL-DNN.

#### Local environment:
Local development is done by using kaggel/python docker image.



