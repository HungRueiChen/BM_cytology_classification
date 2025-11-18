""" create arguments: wsi dir, dest dir, mode (new or retile), create log.txt for successful and fail.txt for failed ones """
#!/usr/bin/env python
# coding: utf-8

# ## load libraries

# In[3]:


import numpy as np
import csv
import random
import cv2
import os
from pathlib import Path
import re
import pandas as pd
import itertools
import argparse
from shutil import rmtree
import json
import matplotlib.pyplot as plt
import time
import argparse

# In[7]:


import openslide
from openslide.deepzoom import DeepZoomGenerator


# ## apply to all slides

# In[4]:
start_time = time.time()

parser = argparse.ArgumentParser(description="Tile WSIs in source_dir and store in dest_dir", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("source_dir", help = 'WSI directory, must contain mrxs files and corresponding .dat folders')
parser.add_argument("dest_dir", help = 'directory to store tiled images, folders will be created with _tile suffix')
parser.add_argument("--mode", choices = ['new', 'retile'], help = 'new to tile WSIs not recorded in any .txt file; retile to tile WSIs recorded in fail.txt')
args = parser.parse_args()

parent_dir = Path(args.source_dir)
assert parent_dir.exists(), f"Source directory {parent_dir} does not exist."
destination_dir = Path(args.dest_dir)
if not destination_dir.exists():
    os.makedirs(destination_dir)
log_file = destination_dir / 'log.txt'
if not log_file.exists():
    _ = open(log_file, 'w')
fail_file = destination_dir / 'fail.txt'
if not fail_file.exists():
    _ = open(fail_file, 'w')

if args.mode == 'new':
    slide_list = [x.stem for x in parent_dir.iterdir() if x.suffix == '.mrxs']
    print(f'### Found {len(slide_list)} slides in {str(parent_dir)} ###')
    with open(log_file, 'r') as f:
        finished_list = [line.rstrip('\n') for line in f]
    with open(fail_file, 'r') as f:
        failed_list = [line.rstrip('\n') for line in f]
    slide_list = list(set(slide_list) - set(finished_list) - set(failed_list))
    
    
    if len(slide_list) == 0:
        print(f'### All slides have been tiled ###')
        exit()
    else:
        print(f'### {len(slide_list)} left after excluding processed tiles ###')
    
elif args.mode == 'retile':
    with open(fail_file, 'r') as f:
        failed_list = [line.rstrip('\n') for line in f]
    with open(log_file, 'r') as f:
        finished_list = [line.rstrip('\n') for line in f]
    slide_list = list(set(failed_list) - set(finished_list))
    
    if len(slide_list) == 0:
        print(f'### No slides have failed ###')
        exit()
    else:
        print(f'### {len(slide_list)} failed previously, retry again ###')

random.shuffle(slide_list)

# In[5]:


# create function to name by absolute pixel, excluding: existing, rectangle, white/black vertices
def single_slide_tiling(slide_path):
    tiles_folder = destination_dir / (slide_path.stem + '_tiles')
    if not tiles_folder.exists():
        tiles_folder.mkdir()

    slide = openslide.OpenSlide(str(slide_path))
    tiles = DeepZoomGenerator(slide, 256, 128, True)
    tile_count = tiles.level_tiles[-1]

    for column in range(tile_count[0]):
        for row in range(tile_count[1]):
            x, y = tiles.get_tile_coordinates(tiles.level_count-1, (column, row))[0]
            image_path = tiles_folder / f'{slide_path.stem}_{x}_{y}.png'
            if not image_path.exists():
                tile = np.array(tiles.get_tile(tiles.level_count-1, (column, row)))
                if tile.shape[0] == tile.shape[1]:
                    vertices = [tile[0, 0]==255, tile[0, -1]==255, tile[-1, 0]==255, tile[-1, -1]==255, 
                                tile[0, 0]==0, tile[0, -1]==0, tile[-1, 0]==0, tile[-1, -1]==0]
                    if not np.array(vertices).all(axis = 1).any():
                        img = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(image_path), img)


# In[29]:


for slide_id in slide_list:
    if not os.path.exists(os.path.join(parent_dir, slide_id+'.lck')) and not os.path.exists(os.path.join(parent_dir, slide_id+'.ok')):
        with open(os.path.join(parent_dir, slide_id+'.lck'), 'w', newline='') as writer:
            writer.write(str(time.time()))    
        
        if args.mode == 'retile':
            print(f'### Retrying {slide_id} ###')
            rmtree(destination_dir / f'{slide_id}_tiles')
        else:
            print(f'### Processing {slide_id} ###')
            
        try:
            single_slide_tiling(parent_dir / (slide_id + '.mrxs'))
            os.rename(os.path.join(parent_dir, slide_id+'.lck'), os.path.join(parent_dir, slide_id+'.ok'))
            with open(log_file, 'a') as f:
                f.write(slide_id + '\n')
            print(f'### Finished {slide_id} successfully, saved in {slide_id}_tiles ###')
        except Exception as e:
            if args.mode == 'new':
                with open(fail_file, 'a') as f:
                    f.write(slide_id + '\n')
            print(f'### Failed because of {e} ###')
    
    end_time = time.time()
    print(f'### Time elapsed: {end_time - start_time} seconds ###')