"""
Simplified Classification Training for Ablation Study
Removed SLURM job submission and kept only DenseNet121 architecture.
"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import random
import cv2
import os
from pathlib import Path
import pandas as pd
import argparse
from shutil import rmtree
import json
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Resizing, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.optimizers.legacy import Adam, SGD
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import CSVLogger
from tensorflow_addons.optimizers import RectifiedAdam

from sklearn.utils.class_weight import compute_class_weight

# Arguments
parser = argparse.ArgumentParser(
    description="Train DenseNet121 for disease classification (Ablation Study)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("directory", help='Image directory, must contain subdirectories: training, validation, test')
parser.add_argument("exp_name", help='Experiment name for saving logs and models')
parser.add_argument("--mode", choices=['initiate', 'resume'], default='initiate',
                   help='Start from epoch 1 or resume from last stored model')
parser.add_argument("--optimizer", choices=['SGD', 'Adam', 'rAdam'], default='SGD',
                   help='Optimizer')
parser.add_argument("--batch_size", type=int, default=8,
                   help='Batch size')
parser.add_argument("--learning_rate", type=float, default=0.001,
                   help='Initial learning rate (will be reduced by 10x for fine-tuning)')
parser.add_argument("--freeze_epochs", type=int, default=10,
                   help='Epochs for freezing stage (train only top layers)')
parser.add_argument("--fine_tune_epochs", type=int, default=90,
                   help='Epochs for fine-tune stage (train all layers)')
parser.add_argument("--high_wt", type=float, default=1.0,
                   help='Multiply class weights > 1 by this factor')
parser.add_argument("--log_dir", default=None,
                   help='Directory to store logs and models. Defaults to ~/16tb2/BM_cytology_classification/stage_1_ablation/<exp_name>')
args = parser.parse_args()

# JSON encoder for special data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Define directories
mother_dir = Path(args.directory)
training_dir = mother_dir / 'training'
validation_dir = mother_dir / 'validation'

# Load datasets
train_ds = image_dataset_from_directory(
    training_dir, labels='inferred', label_mode='categorical',
    batch_size=args.batch_size, image_size=(512, 512), follow_links=True
)
val_ds = image_dataset_from_directory(
    validation_dir, labels='inferred', label_mode='categorical',
    batch_size=args.batch_size, image_size=(512, 512), follow_links=True
)

# Compute class weights (fast numpy implementation)
# Extract labels from file paths instead of loading all images
print("Computing class weights...")
class_names = train_ds.class_names
file_paths = train_ds.file_paths

# Get class indices from file paths
classes = []
for file_path in file_paths:
    # File path format: .../class_name/image.png
    class_name = Path(file_path).parent.name
    class_idx = class_names.index(class_name)
    classes.append(class_idx)

classes = np.array(classes)
counts = np.bincount(classes)
total_samples = len(classes)
n_classes = len(counts)

# Balanced weight formula: total_samples / (n_classes * counts)
weights = total_samples / (n_classes * counts)

# Apply high_wt multiplier to weights > 1
class_weight = {}
for i in range(n_classes):
    class_weight[i] = weights[i] * args.high_wt if weights[i] > 1 else weights[i]

print(f"Class weights: {class_weight}")

# Learning rates
freeze_lr = args.learning_rate
fine_tune_lr = freeze_lr * 0.1
fc_node = 256

# Create log directory
if args.log_dir:
    log_dir = Path(args.log_dir).expanduser()
else:
    log_dir = Path('~/16tb2/BM_cytology_classification/stage_1_ablation').expanduser() / args.exp_name
log_dir.mkdir(parents=True, exist_ok=True)

# Model establishment & training configuration
if args.mode == 'initiate':
    print(f"\nInitiating new training: {args.exp_name}")
    print(f"Log directory: {log_dir}\n")
    
    latest_epoch = 0
    best_val_acc = 0
    freeze_epochs = args.freeze_epochs
    fine_tune_epochs = args.fine_tune_epochs
    
    # DenseNet121 architecture
    base_model = DenseNet121(include_top=False, weights='imagenet', pooling='avg')
    preprocess_fn = tf.keras.applications.densenet.preprocess_input
    resize_side = 224
    
    # Build model
    input_layer = Input(shape=(512, 512, 3))
    x = RandomFlip("horizontal")(input_layer)
    x = RandomRotation((-1, 1), fill_mode='constant', fill_value=255)(x)
    x = RandomZoom((-0.1, 0.1), fill_mode='constant', fill_value=255)(x)
    
    x = Resizing(resize_side, resize_side)(x)
    x = preprocess_fn(x)
    x = base_model(x, training=False)
    
    x = Dense(fc_node, activation='relu', name='fc')(x)
    x = Dropout(0.3, name='do')(x)
    output_layer = Dense(5, activation='softmax', name='cl')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

elif args.mode == 'resume':
    print(f"\nResuming training: {args.exp_name}")
    print(f"Log directory: {log_dir}\n")
    
    # Find latest checkpoint
    tf_epoch_list = [int(x.stem[12:]) for x in log_dir.iterdir() 
                     if x.is_dir() and 'checkpoint' in x.stem]
    latest_epoch = max(tf_epoch_list) if tf_epoch_list else 0
    assert latest_epoch > 0, f"No previous models found in {log_dir}, unable to resume."
    
    freeze_epochs = max(0, args.freeze_epochs - latest_epoch)
    fine_tune_epochs = min(args.fine_tune_epochs, 
                          args.fine_tune_epochs + args.freeze_epochs - latest_epoch)
    
    print(f'Loading model: checkpoint_e{latest_epoch}')
    model = load_model(log_dir / f'checkpoint_e{latest_epoch}', compile=True)
    
    # Truncate log.csv to current epoch
    log_df = pd.read_csv(log_dir / 'log.csv', index_col='epoch')
    log_df = log_df.truncate(before=0, after=latest_epoch-1)
    best_val_acc = log_df['val_accuracy'].max()
    log_df.to_csv(log_dir / 'log.csv', mode='w')

# CSV logger
csv_logger = CSVLogger(log_dir / 'log.csv', append=True)

# Stage 1: Train top layer only (freeze base model)
if freeze_epochs > 0:
    print(f"\n{'='*60}")
    print(f"STAGE 1: Freeze Training ({freeze_epochs} epochs)")
    print(f"Learning rate: {freeze_lr}")
    print(f"{'='*60}\n")
    
    # Initialize if no model is loaded
    if freeze_epochs == args.freeze_epochs:
        base_model.trainable = False
        
        if args.optimizer == 'SGD':
            optimizer_config = SGD(learning_rate=freeze_lr)
        elif args.optimizer == 'Adam':
            optimizer_config = Adam(learning_rate=freeze_lr)
        elif args.optimizer == 'rAdam':
            optimizer_config = RectifiedAdam(learning_rate=freeze_lr)
        
        model.compile(loss='categorical_crossentropy', optimizer=optimizer_config, 
                     metrics=['accuracy'])
    
    model.summary()
    
    for epoch in range(latest_epoch, args.freeze_epochs):
        print(f"\nEpoch {epoch + 1}/{args.freeze_epochs}")
        history = model.fit(train_ds, validation_data=val_ds, epochs=epoch + 1, 
                          initial_epoch=epoch, class_weight=class_weight, 
                          callbacks=[csv_logger])
        
        if history.history['val_accuracy'][0] > best_val_acc:
            model.save(log_dir / f'best_e{epoch + 1}', save_format='tf')
            best_val_acc = history.history['val_accuracy'][0]
            print(f"New best validation accuracy: {best_val_acc:.4f}")
        
        # Save checkpoint
        model.save(log_dir / f'checkpoint_e{epoch + 1}', save_format='tf')
        latest_epoch = epoch + 1

# Stage 2: Unfreeze all layers
if fine_tune_epochs > 0:
    print(f"\n{'='*60}")
    print(f"STAGE 2: Fine-Tune Training ({fine_tune_epochs} epochs)")
    print(f"Learning rate: {fine_tune_lr}")
    print(f"{'='*60}\n")
    
    # Initialize if no model is loaded
    if fine_tune_epochs == args.fine_tune_epochs:
        model.trainable = True
        
        if args.optimizer == 'SGD':
            optimizer_config = SGD(learning_rate=fine_tune_lr)
        elif args.optimizer == 'Adam':
            optimizer_config = Adam(learning_rate=fine_tune_lr)
        elif args.optimizer == 'rAdam':
            optimizer_config = RectifiedAdam(learning_rate=fine_tune_lr)
        
        model.compile(loss='categorical_crossentropy', optimizer=optimizer_config, 
                     metrics=['accuracy'])
    
    model.summary()
    
    for epoch in range(latest_epoch, args.freeze_epochs + args.fine_tune_epochs):
        print(f"\nEpoch {epoch + 1}/{args.freeze_epochs + args.fine_tune_epochs}")
        history = model.fit(train_ds, validation_data=val_ds, epochs=epoch + 1, 
                          initial_epoch=epoch, class_weight=class_weight, 
                          callbacks=[csv_logger])
        
        if history.history['val_accuracy'][0] > best_val_acc:
            model.save(log_dir / f'best_e{epoch + 1}', save_format='tf')
            best_val_acc = history.history['val_accuracy'][0]
            print(f"New best validation accuracy: {best_val_acc:.4f}")
        
        # Save checkpoint
        model.save(log_dir / f'checkpoint_e{epoch + 1}', save_format='tf')
        latest_epoch = epoch + 1

# Save final model
model.save(log_dir / f'final_e{args.freeze_epochs + args.fine_tune_epochs}', save_format='tf')

print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Models saved in: {log_dir}")
print(f"{'='*60}\n")
