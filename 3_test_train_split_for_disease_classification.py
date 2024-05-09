#!/usr/bin/env python
# coding: utf-8

# ## Import Module

# In[1]:


import numpy as np
import tensorflow as tf
import csv
from PIL import Image
import random
import tables
import cv2
import os
from pathlib import Path
import h5py
import re
import pandas as pd
import itertools
import argparse
from shutil import rmtree, copy
import json
import matplotlib.pyplot as plt
plt.switch_backend('agg')
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, GlobalAveragePooling2D, Concatenate, Input
from tensorflow.keras.layers import Resizing, Rescaling, RandomFlip, RandomRotation, RandomZoom, Lambda
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import backend as k
from tensorflow.keras import regularizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.efficientnet import EfficientNetB5, EfficientNetB0
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
# import imgaug.augmenters as iaa

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


# In[5]:


# save the json file and start copying to directories
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


# ## 0916 Revert CML ALL + Recheck all

# In[25]:
# which WSIs to include as the total cohort

# Classification_20230909.csv has two columns: 'New_Image_ID': ['TV0001', ...] and 'Recheck': ['AML', ...]
df = pd.read_csv('./Bone_marrow_cytology_tiles/Classification_20230909.csv')
original_df = pd.read_csv('./Bone_marrow_cytology_tiles/Classification_20230324.csv')
df = df.loc[df['New_Image_ID'].isin(original_df['New_Image_ID'])]

df_0512 = pd.read_csv('16tb2/202308_BM_archive_local/BM_cytology_tile_select/0512_Slide_check_for_classification.csv', header = None, names = ['New_Image_ID', 'Preserve'])
excluded_list = df_0512.loc[df_0512['Preserve'] == 0]['New_Image_ID'].to_list()

with open('./16tb2/202308_BM_archive_local/BM_cytology_tile_select/0523_VGG16_20e_0001_then_180e_00001_SGD/all_probs/poor_slides.txt', 'r') as f:
    for line in f:
        excluded_list.append(line.rstrip().split('.')[0])


# 0829 poor slides
excluded_list += ['TV0564', 'TV0696']

included_df = df.loc[~df['New_Image_ID'].isin(excluded_list)]
included_df.value_counts(subset = ['Recheck'])
# print(f'Total: {len(included_df)}')


# In[26]:
# split WSIs into training, validation, and test sets in the proportions of 7:1:2

all_group = {'training': dict(), 'validation': dict(), 'test': dict()}

def create_cohort(cls_list, included_df, all_group):
    s_list = []
    for c in cls_list.split('_'):
        s_list += included_df.loc[included_df['Recheck'] == c]['New_Image_ID'].to_list()
    random.shuffle(s_list)
    cutoff1 = int(len(s_list) * 0.7)
    delta = (len(s_list) - cutoff1) / 3.0
    cutoff2 = cutoff1 + int(np.rint(delta))
    all_group['training'][cls_list] = s_list[:cutoff1]
    all_group['validation'][cls_list] = s_list[cutoff1:cutoff2]
    all_group['test'][cls_list] = s_list[cutoff2:]
    
for cls_list in ['ALL', 'AML_APL', 'CML', 'Lymphoma_CLL', 'MM']:
    create_cohort(cls_list, included_df, all_group)


# In[27]:
# store in json format for cohort copying

with open('./0916_cohort.json', 'w') as f:
    json.dump(all_group, f, cls = NumpyEncoder)