import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import csv
from PIL import Image
import random
import cv2
import os
from pathlib import Path
import pandas as pd
import itertools
import argparse
from shutil import rmtree, copy
import json
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.ndimage import zoom
from scipy.interpolate import RBFInterpolator
from scipy.stats import mode
import openslide
from openslide.deepzoom import DeepZoomGenerator

from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, GlobalAveragePooling2D, Concatenate, Input
from tensorflow.keras.layers import Resizing, Rescaling, RandomFlip, RandomRotation, RandomZoom, Lambda
from tensorflow.keras.optimizers.legacy import RMSprop, Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.convnext import ConvNeXtTiny
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow_addons.optimizers import RectifiedAdam
from keras.layers.convolutional.base_conv import Conv

def crop(wsi_thumbnail, margin = 5):
    img = np.array(wsi_thumbnail)
    content_args = np.argwhere(img != 255)
    mins = content_args.min(axis = 0)
    maxes = content_args.max(axis = 0)
    return img[mins[0]-margin:maxes[0]+margin+1, mins[1]-margin:maxes[1]+margin+1]

def show_imgwithheat(img, heatmap, alpha=0.4):
    zoom_factor = (img.shape[0] / heatmap.shape[0], img.shape[1] / heatmap.shape[1])
    resized = zoom(heatmap, zoom_factor)
    colored = np.uint8(cm.jet(resized)[..., :3]*255)
    superimposed_img = colored * alpha + img * (1. - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    return superimposed_img

def create_grid(x_grid, y_grid):
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    grid = np.meshgrid(x_grid[:-1] + dx/2, y_grid[:-1] + dy/2)
    grid = np.flip(np.array(grid).reshape(2, -1).T, axis = 1)
    return grid

with open('0sync/Final_BM_cytology_all_200_8models_m/Tile/densenet121_5e_0005_then_35e_00005_SGD/1202_cohort_tiles.json', 'r') as f:
    tile_cohort = json.load(f)
classes = ['ALL', 'AML_APL', 'CML', 'Lymphoma_CLL', 'MM']
color_list = ['red', 'orange', 'yellow', 'lime', 'cyan']
model = load_model('0sync/Final_BM_cytology_all_200_8models_m/best_models/densenet121_e23')
down_sample = 8
save_dir = Path('Final_BM_cytology_all_200/WSI_softened_map')
os.makedirs(save_dir, exist_ok = True)
_, dummy_ax = plt.subplots(nrows=1, ncols=1, figsize=(1, 1), squeeze = False)
'''
for i, c in enumerate(classes):
    for pid in tile_cohort['test'][c]['ids']:
        if pid in [x.name[:6] for x in save_dir.iterdir()]:
            continue
        if pid == 'TV0815' or pid == 'TV1069':
            continue
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 12), squeeze = False)
        """ tiles """
        tile_list = [x for x in tile_cohort['test'][c]['tiles'] if pid in x]
        img_series = []
        x_coords = []
        y_coords = []
        for tile_name in tile_list:
            tile_dir = Path(f'Bone_marrow_cytology_tiles/{pid}_tiles')
            try:
                img = cv2.imread(str(tile_dir / tile_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_series.append(img)
                parsed_tile_name = tile_name.replace('.', '_').split('_')
                x_coords.append(int(parsed_tile_name[1]))
                y_coords.append(int(parsed_tile_name[2]))
            except:
                continue
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        img_series = np.array(img_series)
        pred_probs = model.predict(img_series)
        pred_classes = pred_probs.argmax(axis = 1)
        point_colors = [color_list[x] for x in pred_classes]
        """ WSI """
        if int(pid[2:6]) >= 1148:
            wsi = openslide.OpenSlide(f'Bone_marrow_cytology_wsi_202308/{pid}.mrxs')
        else:
            wsi = openslide.OpenSlide(f'Bone_marrow_cytology_wsi/{pid}.mrxs')
        wsi_thumbnail = wsi.get_thumbnail((1024, 1024))
        wsi_thumbnail = crop(wsi_thumbnail, margin = -1)
        wsi_width = int(wsi.properties['openslide.bounds-width'])
        wsi_height = int(wsi.properties['openslide.bounds-height'])
        wsi_x0 = int(wsi.properties['openslide.bounds-x'])
        wsi_y0 = int(wsi.properties['openslide.bounds-y'])
        wsi.close()
        norm_x_coords = np.interp(x_coords, [wsi_x0, wsi_x0 + wsi_width], [0, wsi_thumbnail.shape[1]])
        norm_y_coords = np.interp(y_coords, [wsi_y0, wsi_y0 + wsi_height], [0, wsi_thumbnail.shape[0]])
        ax[0, 0].imshow(wsi_thumbnail)
        ax[0, 0].scatter(norm_x_coords, norm_y_coords, c = point_colors, marker = 's')
        # ax[0, 0].set_title(f'Scatter {pid}: {gt}', fontfamily = 'serif', fontsize = 20)
        ax[0, 0].axis('off')

        x_bin = np.round(wsi_thumbnail.shape[1] / down_sample).astype("uint8")
        y_bin = np.round(wsi_thumbnail.shape[0] / down_sample).astype("uint8")
        _, x_grid, y_grid, _ = dummy_ax[0, 0].hist2d(norm_x_coords, norm_y_coords, 
                                          bins = [x_bin, y_bin],
                                          range = [[0, wsi_thumbnail.shape[1]], [0, wsi_thumbnail.shape[0]]])
        grid = create_grid(x_grid, y_grid)

        """ density map at lower resolution """
        density_map, x_grid, y_grid, _ = dummy_ax[0, 0].hist2d(norm_x_coords, norm_y_coords, 
                                          bins = [x_bin//2, y_bin//2],
                                          range = [[0, wsi_thumbnail.shape[1]], [0, wsi_thumbnail.shape[0]]])
        density_map = density_map.T.flatten()
        density_map = np.sqrt(density_map / density_map.max())
        density_grid = create_grid(x_grid, y_grid)
        density_map = RBFInterpolator(density_grid, density_map)(grid)
        density_map = density_map.reshape(y_bin, x_bin)
        density_heatmap = show_imgwithheat(wsi_thumbnail, density_map)
        """ probability map """
        pred_scalar = pred_probs[:, mode(pred_classes)[0][0]]
        pred_scatter = np.array([norm_y_coords, norm_x_coords]).T
        prob_map = RBFInterpolator(pred_scatter, pred_scalar)(grid)
        prob_map = prob_map.reshape(y_bin, x_bin)
        prob_heatmap = show_imgwithheat(wsi_thumbnail, prob_map)
        """ final map """
        assert density_map.shape == prob_map.shape
        salience_map = prob_map * density_map
        salience_heatmap = show_imgwithheat(wsi_thumbnail, salience_map)
        ax[0, 1].imshow(salience_heatmap)
        # ax[0, 1].set_title(f'Salience {pid}: {gt}', fontfamily = 'serif', fontsize = 20)
        ax[0, 1].axis('off')
        
        gt = c.split('_')[0]
        pred = mode(pred_classes, axis = None, keepdims = False).mode
        pred = classes[pred].split('_')[0]
        fig.savefig(save_dir / f'{pid}_{gt}_to_{pred}.png', dpi = fig.dpi, bbox_inches = 'tight')
'''
pid_list = ['TV0335', 'TV0167', 'TV0972', 'TV0756', 'TV0458']
for i, c in enumerate(classes):
    pid = pid_list[i]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 4), squeeze = False)
    """ tiles """
    tile_list = [x for x in tile_cohort['test'][c]['tiles'] if pid in x]
    img_series = []
    x_coords = []
    y_coords = []
    for tile_name in tile_list:
        tile_dir = Path(f'Bone_marrow_cytology_tiles/{pid}_tiles')
        try:
            img = cv2.imread(str(tile_dir / tile_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_series.append(img)
            parsed_tile_name = tile_name.replace('.', '_').split('_')
            x_coords.append(int(parsed_tile_name[1]))
            y_coords.append(int(parsed_tile_name[2]))
        except:
            continue
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    img_series = np.array(img_series)
    pred_probs = model.predict(img_series)
    pred_classes = pred_probs.argmax(axis = 1)
    point_colors = [color_list[x] for x in pred_classes]
    """ WSI """
    if int(pid[2:6]) >= 1148:
        wsi = openslide.OpenSlide(f'Bone_marrow_cytology_wsi_202308/{pid}.mrxs')
    else:
        wsi = openslide.OpenSlide(f'Bone_marrow_cytology_wsi/{pid}.mrxs')
    wsi_thumbnail = wsi.get_thumbnail((1024, 1024))
    wsi_thumbnail = crop(wsi_thumbnail, margin = -1)
    wsi_width = int(wsi.properties['openslide.bounds-width'])
    wsi_height = int(wsi.properties['openslide.bounds-height'])
    wsi_x0 = int(wsi.properties['openslide.bounds-x'])
    wsi_y0 = int(wsi.properties['openslide.bounds-y'])
    wsi.close()
    norm_x_coords = np.interp(x_coords, [wsi_x0, wsi_x0 + wsi_width], [0, wsi_thumbnail.shape[1]])
    norm_y_coords = np.interp(y_coords, [wsi_y0, wsi_y0 + wsi_height], [0, wsi_thumbnail.shape[0]])
    transposed_wsi = np.swapaxes(wsi_thumbnail, 0, 1)
    ax[0, 0].imshow(transposed_wsi)
    ax[0, 0].scatter(norm_y_coords, norm_x_coords, c = point_colors, marker = 's')
    # ax[0, 0].set_title(f'Scatter {pid}: {gt}', fontfamily = 'serif', fontsize = 20)
    ax[0, 0].axis('off')

    x_bin = np.round(wsi_thumbnail.shape[1] / down_sample).astype("uint8")
    y_bin = np.round(wsi_thumbnail.shape[0] / down_sample).astype("uint8")
    _, x_grid, y_grid, _ = dummy_ax[0, 0].hist2d(norm_x_coords, norm_y_coords, 
                                      bins = [x_bin, y_bin],
                                      range = [[0, wsi_thumbnail.shape[1]], [0, wsi_thumbnail.shape[0]]])
    grid = create_grid(x_grid, y_grid)

    """ density map at lower resolution """
    density_map, x_grid, y_grid, _ = dummy_ax[0, 0].hist2d(norm_x_coords, norm_y_coords, 
                                      bins = [x_bin//2, y_bin//2],
                                      range = [[0, wsi_thumbnail.shape[1]], [0, wsi_thumbnail.shape[0]]])
    density_map = density_map.T.flatten()
    density_map = np.sqrt(density_map / density_map.max())
    density_grid = create_grid(x_grid, y_grid)
    density_map = RBFInterpolator(density_grid, density_map)(grid)
    density_map = density_map.reshape(y_bin, x_bin)
    density_heatmap = show_imgwithheat(wsi_thumbnail, density_map)
    """ probability map """
    pred_scalar = pred_probs[:, mode(pred_classes)[0][0]]
    pred_scatter = np.array([norm_y_coords, norm_x_coords]).T
    prob_map = RBFInterpolator(pred_scatter, pred_scalar)(grid)
    prob_map = prob_map.reshape(y_bin, x_bin)
    prob_heatmap = show_imgwithheat(wsi_thumbnail, prob_map)
    """ final map """
    assert density_map.shape == prob_map.shape
    salience_map = prob_map * density_map
    salience_heatmap = show_imgwithheat(wsi_thumbnail, salience_map)
    transposed_salience_heatmap = np.swapaxes(salience_heatmap, 0, 1)
    ax[0, 1].imshow(transposed_salience_heatmap)
    # ax[0, 1].set_title(f'Salience {pid}: {gt}', fontfamily = 'serif', fontsize = 20)
    ax[0, 1].axis('off')

    gt = c.split('_')[0]
    pred = mode(pred_classes, axis = None, keepdims = False).mode
    pred = classes[pred].split('_')[0]
    fig.savefig(save_dir / f'transposed_{pid}_{gt}_to_{pred}.png', dpi = fig.dpi, bbox_inches = 'tight')