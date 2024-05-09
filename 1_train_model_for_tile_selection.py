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


batch_size = 8
resize_side = 224
lr = 0.001
freeze_epochs = 20
finetune_epochs = 180

# datasets
training_dir = Path(f'../BM_cytology_tile_select/0523_training')
training_ds = image_dataset_from_directory(training_dir, batch_size = batch_size, image_size = (512, 512))

validation_dir = Path(f'../BM_cytology_tile_select/0523_validation')
validation_ds = image_dataset_from_directory(validation_dir, batch_size = batch_size, image_size = (512, 512))

log_dir = Path(f'../BM_cytology_tile_select/0523_VGG16_{freeze_epochs}e_0001_then_{finetune_epochs}e_00001_SGD')
if log_dir.exists():
    rmtree(log_dir)
os.makedirs(log_dir, exist_ok = True)

# calculate weight
good_count = 0
total_count = 0

for i, batch in enumerate(training_ds):
    labels = np.array(batch[1])
    total_count += labels.shape[0]
    good_count += labels.sum()

poor_count = total_count - good_count
print(f'Training set good counts:{good_count}, poor counts:{poor_count}')

weight_for_1 = (1 / good_count) * (total_count / 2.0)
weight_for_0 = (1 / poor_count) * (total_count / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}
print(class_weight)

# model establishment
input_layer = Input(shape = (512, 512, 3))
x = RandomFlip("horizontal")(input_layer)
x = RandomRotation((-1, 1), fill_mode = 'constant', fill_value = 255)(x)
x = RandomZoom((-0.1, 0.1), fill_mode = 'constant', fill_value = 255)(x)

x = Resizing(resize_side, resize_side)(x)
x = tf.keras.applications.vgg16.preprocess_input(x)

base_model = VGG16(include_top = False, weights = 'imagenet', pooling = 'avg')
x = base_model(x, training = False)

x = Dense(256, activation = 'relu', name = 'fc')(x)
x = Dropout(0.3, name = 'do')(x)
output_layer = Dense(1, activation = 'sigmoid', name = 'cl')(x)
model = Model(inputs = input_layer, outputs = output_layer)

checkpoint = ModelCheckpoint(filepath = str(log_dir)+'/e{epoch}.h5', monitor = 'val_accuracy', save_best_only = True)

# stage 1
base_model.trainable = False
model.compile(loss = 'binary_crossentropy', optimizer = SGD(learning_rate = lr), metrics=['accuracy'])
model.summary()

metrics_hx = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
history = model.fit(training_ds, validation_data = validation_ds, epochs = freeze_epochs, class_weight = class_weight, callbacks = [checkpoint])
for k, v in metrics_hx.items():
    v += history.history[k]
model.save(log_dir / f'final_e{freeze_epochs}.h5')

# stage 2
base_model.trainable = True
model.compile(loss = 'binary_crossentropy', optimizer = SGD(learning_rate = 0.1*lr), metrics=['accuracy'])
model.summary()

history = model.fit(training_ds, validation_data = validation_ds, epochs = freeze_epochs + finetune_epochs, 
                    class_weight = class_weight, callbacks = [checkpoint], initial_epoch = freeze_epochs)
for k, v in metrics_hx.items():
    v += history.history[k]
model.save(log_dir / f'final_e{freeze_epochs + finetune_epochs}.h5')
with open(log_dir / 'log.json', 'w') as file:
    json.dump(metrics_hx, file)
    
# learning curve
loss = metrics_hx['loss']
val_loss = metrics_hx['val_loss']

acc = metrics_hx['accuracy']
val_acc = metrics_hx['val_accuracy']

epochs_range = range(len(loss))
learning_curve = plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right', fontsize = 10)
plt.title('Training and Validation Loss', fontsize = 15)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right', fontsize = 10)
plt.title('Training and Validation Accuracy', fontsize = 15)

learning_curve.savefig(log_dir / 'learning_curve.png', bbox_inches = 'tight')
    
# testing
epoch_list = [int(x.stem[1:]) for x in log_dir.glob('*.h5') if not 'final' in x.stem]
model = load_model(log_dir / f'e{max(epoch_list)}.h5')

test_dir = Path(f'../BM_cytology_tile_select/0523_test')
test_ds = image_dataset_from_directory(test_dir, batch_size = batch_size, image_size = (512, 512), shuffle = False)

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds)
y_pred_binary = y_pred[:, 0] >= 0.5

correct = np.equal(y_true, y_pred_binary)
test_acc = sum(correct) / len(correct)

# confusion matrix
cm = confusion_matrix(y_true, y_pred_binary)
cm_norm = confusion_matrix(y_true, y_pred_binary, normalize = 'true')

classes = ['poor', 'good']
fig_cm = plt.figure(figsize = (10, 5))
for i, mtx in enumerate([cm, cm_norm.round(3)]):
    plt.subplot(1, 2, i+1)
    plt.imshow(mtx, interpolation = 'nearest', vmin = 0, cmap = plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0, fontsize = 12)
    plt.yticks(tick_marks, classes, fontsize = 12)
    thresh = mtx.max() / 2.
    for i, j in itertools.product(range(mtx.shape[0]), range(mtx.shape[1])):
        plt.text(j, i, mtx[i, j], horizontalalignment = "center", fontsize = 25, color = "white" if mtx[i, j] > thresh else "black")
    plt.ylabel('Groundtruth', fontsize = 12)
    plt.xlabel('Prediction', fontsize = 12)
plt.suptitle(f'Tile selection: {log_dir.name}, e{max(epoch_list)}', fontsize = 15)
fig_cm.savefig(log_dir / 'confusion_matrix.png', dpi = fig_cm.dpi, bbox_inches = 'tight') 
plt.close()

# roc curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
area = auc(fpr, tpr)

fig_roc = plt.figure(figsize = (5.6, 5.6))
plt.plot(fpr, tpr, label = 'area = {0:0.3f}'.format(area), color = 'navy', linewidth = 2)
plt.plot([0, 1], [0, 1], 'k--', lw = 1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 12)
plt.ylabel('True Positive Rate', fontsize = 12)
plt.title(f'Tile selection: {log_dir.name}, e{max(epoch_list)}', fontsize = 12)
plt.legend(loc = "lower right", fontsize = 12)

fig_roc.savefig(log_dir / 'ROC_curve.png', dpi = fig_cm.dpi, bbox_inches = 'tight')
plt.close()

# test results
test_result = {'AUC': area, 'test_acc': test_acc, 'confusion_matrix': cm, 'normalized_cm': cm_norm}

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open(log_dir / 'test_result.json', 'w') as f:
    json.dump(test_result, f, cls = NumpyEncoder)