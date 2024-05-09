import numpy as np
import tensorflow as tf
from PIL import Image
import random
import cv2
import os
from pathlib import Path
import pandas as pd
import itertools
import argparse
from shutil import rmtree, copy
from copy import deepcopy
import json
import matplotlib.pyplot as plt

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
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, matthews_corrcoef, cohen_kappa_score

parent_dir = Path('../Final_BM_cytology_all_200')

# load cohort data from json
with open(parent_dir / '1202_cohort.json', 'r') as f:
    group_1202 = json.load(f)
cohort_1202 = deepcopy(group_1202)

for group in ['training', 'validation', 'test']:
    for label in ['ALL', 'AML_APL', 'CML', 'Lymphoma_CLL', 'MM']:
        cohort_1202[group][label] = {'ids': cohort_1202[group][label], 'tiles': []}
        for slide_id in group_1202[group][label]:
            df = pd.read_csv(f'../BM_cytology_tile_select/0523_VGG16_20e_0001_then_180e_00001_SGD/all_probs/{slide_id}.csv')
            tile_folder = Path(f'../Bone_marrow_cytology_tiles/{slide_id}_tiles/')
            os.makedirs(parent_dir / group / label, exist_ok = True)
            
            if slide_id == 'TV0577':
                excluded_list = [f'../Bone_marrow_cytology_tiles/TV0577_tiles/{x}' for x in ['TV0577_68860_175674.png', 'TV0577_67836_167226.png']]
                df = df.loc[~df['Path'].isin(excluded_list)]
            
            
            if df.loc[df['Prob'] >= 0.8].shape[0] < 200:
                # fewer than 200 tiles have good quality score > 0.8, include tiles whose scores are between 0.5 and 0.8
                chosen1 = df.loc[df['Prob'] >= 0.8]
                remaining = 200 - len(chosen1)
                chosen2 = df.loc[(df['Prob'] < 0.8) & (df['Prob'] >= 0.5)].sample(n = remaining)
                chosen = pd.concat([chosen1, chosen2])
            else:
                # more than 200 tiles have good quality score > 0.8, randomly select 200 from them as ROIs
                chosen = df.loc[df['Prob'] >= 0.8].sample(n = 200)
            
            for i, row in chosen.iterrows():
                tile = Path(row['Path']).name
                try:
                    copy(tile_folder / tile, parent_dir / group / label / tile)
                    cohort_1202[group][label]['tiles'].append(tile)
                except Exception as e:
                    print(tile, e)
# store included ROIs in another json file
with open(parent_dir / '1202_cohort_tiles.json', 'w') as f:
    json.dump(cohort_1202, f)