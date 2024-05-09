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

from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.ndimage import zoom

# arguments
parser = argparse.ArgumentParser(description="Visualize Performance of Best Model on Given Directory with gradCAM", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("directory", help = 'image directory, must contain a test folder and at least one model')
parser.add_argument("--architecture", choices = ['vgg16', 'inceptionv3', 'resnet50', 'densenet121', 'xception', 'mobilenetv3large', 'inceptionresnetv2', 'nasnetmobile', 'convnexttiny', 'efficientnetv2b3'], default = 'vgg16', help = 'CNN backbbone architecture')
parser.add_argument("--optimizer", choices = ['SGD', 'Adam', 'RMSprop', 'rAdam'], default = 'SGD', help = 'optimizer')
parser.add_argument("--batch_size", type = int, default = 8, help = 'batch size')
parser.add_argument("--freeze_lr", type = str, default = '0.001', help = 'learning rate when base model is freezed at first, fine tune lr will be 1/10 of freeze lr')
parser.add_argument("--freeze_epochs", type = int, default = 10, help = 'epochs for freezing stage')
parser.add_argument("--fine_tune_epochs", type = int, default = 90, help = 'epochs for fine tune stage')
parser.add_argument("-l", "--layers", action = 'append', type = str, default = None, help = 'layer name(s) in architecture to produce gradCAM')
parser.add_argument("--sample_size", type = int, default = 10, help = 'number of tiles per class to visualize')
args = parser.parse_args()

# set vars
classes = ['ALL', 'AML_APL', 'CML', 'Lymphoma_CLL', 'MM']
classes_for_display = ['ALL', 'AML', 'CML', 'Lymphoma', 'MM']
mother_dir = Path('../') / args.directory
test_dir = mother_dir / 'test'

freeze_lr = float(args.freeze_lr)
fine_tune_lr = freeze_lr * 0.1
fr_lr_str = args.freeze_lr.replace('.', '')
tu_lr_str = '0' + fr_lr_str
log_dir = Path(mother_dir / f'{args.architecture}_{args.freeze_epochs}e_{fr_lr_str}_then_{args.fine_tune_epochs}e_{tu_lr_str}_{args.optimizer}')
if not log_dir.exists():
    raise
vis_dir = log_dir / 'gradCAM'
for c in classes_for_display:
    os.makedirs(vis_dir / c, exist_ok = True)

# select the best model
log_df = pd.read_csv(log_dir / 'log.csv', index_col = 'epoch')
best_epoch = log_df['val_accuracy'].idxmax() + 1
print(f'Loading model: e{best_epoch}')
old_model = load_model(log_dir / f'best_e{best_epoch}')

# rebuild model by un-nesting the submodel
input_layer = Input(shape = (512, 512, 3))
x = RandomFlip("horizontal")(input_layer)
x = RandomRotation((-1, 1), fill_mode = 'constant', fill_value = 255)(x)
x = RandomZoom((-0.1, 0.1), fill_mode = 'constant', fill_value = 255)(x)

if args.architecture == 'vgg16':
    preprocess_fn = tf.keras.applications.vgg16.preprocess_input
    resize_side = 224
    x = Resizing(resize_side, resize_side)(x)
    x = preprocess_fn(x)
    x = VGG16(input_tensor = x, include_top = False, weights = 'imagenet', pooling = 'avg')
elif args.architecture == 'inceptionv3':
    preprocess_fn = tf.keras.applications.inception_v3.preprocess_input
    resize_side = 229
    x = Resizing(resize_side, resize_side)(x)
    x = preprocess_fn(x)
    x = InceptionV3(input_tensor = x, include_top = False, weights = 'imagenet', pooling = 'avg')
elif args.architecture == 'resnet50':
    preprocess_fn = tf.keras.applications.resnet.preprocess_input
    resize_side = 224
    x = Resizing(resize_side, resize_side)(x)
    x = preprocess_fn(x)
    x = ResNet50(input_tensor = x, include_top = False, weights = 'imagenet', pooling = 'avg')
elif args.architecture == 'densenet121':
    preprocess_fn = tf.keras.applications.densenet.preprocess_input
    resize_side = 224
    x = Resizing(resize_side, resize_side)(x)
    x = preprocess_fn(x)
    x = DenseNet121(input_tensor = x, include_top = False, weights = 'imagenet', pooling = 'avg')
elif args.architecture == 'xception':
    preprocess_fn = tf.keras.applications.xception.preprocess_input
    resize_side = 299
    x = Resizing(resize_side, resize_side)(x)
    x = preprocess_fn(x)
    x = Xception(input_tensor = x, include_top = False, weights = 'imagenet', pooling = 'avg')
elif args.architecture == 'mobilenetv3large':
    preprocess_fn = tf.keras.applications.mobilenet_v3.preprocess_input
    resize_side = 224
    x = Resizing(resize_side, resize_side)(x)
    x = preprocess_fn(x)
    x = MobileNetV3Large(input_tensor = x, include_top = False, weights = 'imagenet', pooling = 'avg', alpha = 0.75)
elif args.architecture == 'inceptionresnetv2':
    preprocess_fn = tf.keras.applications.inception_resnet_v2.preprocess_input
    resize_side = 299
    x = Resizing(resize_side, resize_side)(x)
    x = preprocess_fn(x)
    x = InceptionResNetV2(input_tensor = x, include_top = False, weights = 'imagenet', pooling = 'avg')
elif args.architecture == 'nasnetmobile':
    preprocess_fn = tf.keras.applications.nasnet.preprocess_input
    resize_side = 224
    x = Resizing(resize_side, resize_side)(x)
    x = preprocess_fn(x)
    x = NASNetMobile(input_tensor = x, include_top = False, weights = 'imagenet', pooling = 'avg', input_shape = (224, 224, 3))
elif args.architecture == 'convnexttiny':
    preprocess_fn = tf.keras.applications.convnext.preprocess_input
    resize_side = 224
    x = Resizing(resize_side, resize_side)(x)
    x = preprocess_fn(x)
    x = ConvNeXtTiny(input_tensor = x, include_top = False, weights = 'imagenet', pooling = 'avg')
elif args.architecture == 'efficientnetv2b3':
    preprocess_fn = tf.keras.applications.efficientnet_v2.preprocess_input
    resize_side = 224
    x = Resizing(resize_side, resize_side)(x)
    x = preprocess_fn(x)
    x = EfficientNetV2B3(input_tensor = x, include_top = False, weights = 'imagenet', pooling = 'avg')
else:
    raise

x.trainable = False
x = Dense(256, activation = 'relu', name = 'fc')(x.output)
x = Dropout(0.3, name = 'do')(x)
output_layer = Dense(5, activation = 'sigmoid', name = 'cl')(x)
model = Model(inputs = input_layer, outputs = output_layer)
model.summary()

# check if provided layers are valid
for layer in args.layers:
    assert layer in [l.name for l in model.layers], f'layer {layer} not in model.layers'

# assign weights from old to new model
delta = 0
for idx, layer in enumerate(old_model.layers):
    if isinstance(layer, tf.keras.models.Model):
        for sub_idx, sub_layer in enumerate(layer.layers[1:]):
            assert type(sub_layer) == type(model.layers[idx + sub_idx]), f'{type(sub_layer)} {type(model.layers[idx + sub_idx])}'
            model.layers[idx + sub_idx].set_weights(sub_layer.get_weights())
            delta += 1
        delta -= 1
        continue
    assert type(old_model.layers[idx]) == type(model.layers[idx + delta]), f'{type(old_model.layers[idx])} {type(model.layers[idx + delta])}'
    model.layers[idx + delta].set_weights(layer.get_weights())

# define functions for gradCAM visualization
def score_function(output, gt_list):
    return [output[i][c] for i, c in enumerate(gt_list)]

def find_layer(model, condition, offset=None, reverse=True) -> tf.keras.layers.Layer:
    found_offset = offset is None
    for layer in reversed(model.layers):
        if not found_offset:
            found_offset = (layer == offset)
        if condition(layer) and found_offset:
            return layer
        if isinstance(layer, tf.keras.Model):
            if found_offset:
                result = find_layer(layer, condition, offset=None, reverse=reverse)
            else:
                result = find_layer(layer, condition, offset=offset, reverse=reverse)
            if result is not None:
                return result
    return None

def gradCAM(model, img, layer_name='last conv layer', category_ids=None):
    img_tensor = np.expand_dims(img, axis=0) if len(img.shape) < 4 else img

    conv_layer = find_layer(model, lambda _l: isinstance(_l, Conv)) if layer_name == 'last conv layer' else model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img_tensor, training = False)
        if category_ids is None:
            category_ids = np.argmax(predictions, axis = 1)
        output = score_function(predictions, category_ids)
        grads = gtape.gradient(output, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2), keepdims = True)
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap, axis = (1, 2), keepdims = True)
    heatmap = np.divide(heatmap, max_heat, out = np.zeros_like(heatmap), where = (max_heat != 0))

    return heatmap

def gradCAMplus(model, img, layer_name='last conv layer', label_name=['ALL', 'AML_APL', 'CML', 'Lymphoma_CLL', 'MM'], category_ids=None):
    img_tensor = np.expand_dims(img, axis=0) if len(img.shape) < 4 else img

    conv_layer = find_layer(model, lambda _l: isinstance(_l, Conv)) if layer_name == 'last conv layer' else model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])
    
    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor, training = False)
                if category_ids is None:
                    category_ids = np.argmax(predictions, axis = 1)
                output = score_function(predictions, category_ids)
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = K.sum(conv_output, axis=tuple(np.arange(len(conv_output.shape))[1:-1]), keepdims = True)

    alpha_num = conv_second_grad
    alpha_denom = conv_second_grad*2.0 + conv_third_grad*global_sum
    alpha_denom = alpha_denom + tf.cast((conv_second_grad == 0.0), conv_second_grad.dtype)
    # alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)
    alphas = alpha_num/alpha_denom
    
    alpha_normalization_constant = K.sum(alphas, axis=tuple(np.arange(len(alphas.shape))[1:-1]), keepdims = True)
    alpha_normalization_constant = alpha_normalization_constant + tf.cast((alpha_normalization_constant == 0.0),
                                                                          alpha_normalization_constant.dtype)
    alphas /= alpha_normalization_constant

    weights = tf.math.maximum(conv_first_grad, 0.0)
    deep_linearization_weights = weights*alphas
    deep_linearization_weights = K.sum(deep_linearization_weights, 
                                       axis=tuple(np.arange(len(deep_linearization_weights.shape))[1:-1]), keepdims = True)
    grad_cam_map = K.sum(deep_linearization_weights*conv_output, axis=-1)
    
    heatmap = np.maximum(grad_cam_map, 0)
    max_heat = np.max(heatmap, axis = (1, 2), keepdims = True)
    heatmap = np.divide(heatmap, max_heat, out = np.zeros_like(heatmap), where = (max_heat != 0))

    return heatmap

def apply_to_img(imgs, heatmap, alpha=0.4):
    zoom_factor = (imgs.shape[1] / heatmap.shape[1], imgs.shape[2] / heatmap.shape[2])
    resized = [zoom(h, zoom_factor) for h in heatmap]
    colored = np.array([np.uint8(cm.jet(r)[..., :3]*255) for r in resized])
    superimposed_imgs = colored * alpha + imgs * (1. - alpha)
    superimposed_imgs = np.clip(superimposed_imgs, 0, 255).astype("uint8")

    return superimposed_imgs

# iterate over test set and plot
for c, disease in enumerate(classes):
    class_dir = test_dir / disease
    sample = random.sample([img for img in class_dir.iterdir()], args.sample_size)
    imgs = []
    for img_path in sample:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    imgs = np.array(imgs)
    results = model.predict(imgs)
    
    graph_list = [imgs]
    graph_name_list = ['original']
    if not args.layers:
        args.layers = ['last conv layer']
    for layer in args.layers:
        cam = gradCAM(model, imgs, layer_name = layer)
        cam = apply_to_img(imgs, cam, alpha = 0.6)
        graph_list.append(cam)
    graph_name_list += args.layers
        
    for i, img in enumerate(imgs):
        fig_cm, ax = plt.subplots(nrows=1, ncols=len(graph_list), figsize=(5*len(graph_list), 5), squeeze = False)
        for g, graph in enumerate(graph_list):
            ax[0, g].imshow(graph[i])
            # ax[0, g].set_title(graph_name_list[g], fontsize = 12, fontfamily = 'serif')
            ax[0, g].axis('off')
        display_disease = classes_for_display[c]
        predicted_disease = classes_for_display[results[i].argmax()]
        # fig_cm.suptitle(f'{sample[i].stem}    gt: {display_disease}    predicted: {predicted_disease}', fontsize = 15, fontfamily = 'serif')
        fig_cm.savefig(vis_dir / display_disease / f'{sample[i].stem}_{display_disease}_to_{predicted_disease}.png', dpi = fig_cm.dpi, bbox_inches = 'tight')
        