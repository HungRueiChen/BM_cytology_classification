"""
Random Tile Selection for Ablation Study
Directly tiles WSI files using OpenSlide and randomly selects 200 valid tiles.
Processes one WSI at a time to manage storage constraints.
"""

import numpy as np
import random
import cv2
import os
from pathlib import Path
import argparse
import json
from copy import deepcopy
import openslide
from openslide.deepzoom import DeepZoomGenerator

def is_valid_tile(tile):
    """
    Check if a tile is valid (not background).
    Excludes tiles where all corner pixels are white (#FFF) or black (#000).
    
    Args:
        tile: numpy array of the tile image
        
    Returns:
        Boolean indicating if tile is valid
    """
    if tile.shape[0] != tile.shape[1]:
        return False
    
    # Check corner pixels for all white or all black
    vertices = [
        tile[0, 0] == 255, tile[0, -1] == 255, tile[-1, 0] == 255, tile[-1, -1] == 255,
        tile[0, 0] == 0, tile[0, -1] == 0, tile[-1, 0] == 0, tile[-1, -1] == 0
    ]
    
    # If any of the corners are white OR any of the corners are black, it's background
    if np.array(vertices).all(axis=1).any():
        return False
    
    return True

def tile_and_select_random(slide_path, num_tiles=200, excluded_tiles=None):
    """
    Tile a WSI and randomly select valid tiles.
    Optimization: Only stores metadata during scanning to save memory.
    
    Args:
        slide_path: Path to the .mrxs WSI file
        num_tiles: Number of tiles to randomly select
        excluded_tiles: List of tile names to exclude (for specific slides)
        
    Returns:
        List of tuples: (tile_name, tile_image, coordinates)
    """
    if excluded_tiles is None:
        excluded_tiles = []
    
    slide = openslide.OpenSlide(str(slide_path))
    tiles = DeepZoomGenerator(slide, 512, 256, True)
    tile_count = tiles.level_tiles[-1]
    
    # First pass: collect all valid tile metadata (indices and coords)
    valid_tile_metadata = []
    slide_id = slide_path.stem
    
    print(f"Scanning {tile_count[0] * tile_count[1]} potential tiles...")
    
    level = tiles.level_count - 1
    
    for column in range(tile_count[0]):
        for row in range(tile_count[1]):
            x, y = tiles.get_tile_coordinates(level, (column, row))[0]
            tile_name = f'{slide_id}_{x}_{y}.png'
            
            # Skip excluded tiles
            if tile_name in excluded_tiles:
                continue
            
            # Get the tile to check validity
            # Note: We retrieve it, check it, then discard it immediately
            tile = np.array(tiles.get_tile(level, (column, row)))
            
            # Check if valid (not background)
            if is_valid_tile(tile):
                # Store only metadata: (tile_name, column, row, x, y)
                valid_tile_metadata.append((tile_name, column, row, x, y))
                
            # Explicitly delete tile reference just in case, though scope exit handles it
            del tile
    
    print(f"Found {len(valid_tile_metadata)} valid tiles")
    
    # Randomly select metadata
    if len(valid_tile_metadata) < num_tiles:
        print(f"Warning: Only {len(valid_tile_metadata)} valid tiles available, less than requested {num_tiles}")
        selected_metadata = valid_tile_metadata
    else:
        selected_metadata = random.sample(valid_tile_metadata, num_tiles)
        
    # Second pass: Retrieve images for selected tiles
    selected_tiles = []
    print(f"Retrieving images for {len(selected_metadata)} selected tiles...")
    
    for tile_name, col, row, x, y in selected_metadata:
        try:
            tile_image = np.array(tiles.get_tile(level, (col, row)))
            selected_tiles.append((tile_name, tile_image, (x, y)))
        except Exception as e:
            print(f"Error retrieving tile {tile_name}: {e}")
    
    return selected_tiles

def process_single_wsi(slide_id, label, group, wsi_source_dir, dest_parent_dir, 
                       num_tiles=200, excluded_tiles=None):
    """
    Process a single WSI: tile it, select random tiles, and save only selected ones.
    
    Args:
        slide_id: WSI identifier
        label: Disease label (ALL, AML_APL, CML, Lymphoma_CLL, MM)
        group: Dataset split (training, validation, test)
        wsi_source_dir: Source directory containing .mrxs WSI files
        dest_parent_dir: Destination parent directory
        num_tiles: Number of tiles to select
        excluded_tiles: List of tile names to exclude
        
    Returns:
        List of selected tile names
    """
    slide_path = Path(wsi_source_dir) / f'{slide_id}.mrxs'
    
    if not slide_path.exists():
        print(f"Error: WSI file {slide_path} does not exist")
        return []
    
    # Create destination directory
    dest_dir = Path(dest_parent_dir) / group / label
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"\nProcessing {slide_id}...")
    
    # Tile and randomly select
    selected_tiles = tile_and_select_random(slide_path, num_tiles, excluded_tiles)
    selected_tile_names = []
    
    # Save selected tiles
    print(f"Saving {len(selected_tiles)} selected tiles...")
    for tile_name, tile_img, coords in selected_tiles:
        try:
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
            output_path = dest_dir / tile_name
            cv2.imwrite(str(output_path), img)
            selected_tile_names.append(tile_name)
        except Exception as e:
            print(f"Error saving {tile_name}: {e}")
    
    print(f"Saved {len(selected_tile_names)} tiles for {slide_id} to {dest_dir}")
    
    return selected_tile_names

def main():
    parser = argparse.ArgumentParser(
        description="Tile WSI and randomly select 200 tiles for ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--slide_id", help="WSI identifier (e.g., TV0001)")
    parser.add_argument("--label", help="Disease label (ALL, AML_APL, CML, Lymphoma_CLL, MM)")
    parser.add_argument("--group", help="Dataset split (training, validation, test)")
    parser.add_argument("--wsi_source_dir", default="../Bone_marrow_cytology_WSI",
                        help="Source directory containing .mrxs WSI files")
    parser.add_argument("--dest_parent_dir", default="../Final_BM_cytology_all_200_random",
                        help="Destination parent directory for selected tiles")
    parser.add_argument("--num_tiles", type=int, default=200,
                        help="Number of tiles to randomly select")
    parser.add_argument("--cohort_json", default=None,
                        help="Path to cohort JSON file for batch processing")
    parser.add_argument("--process_all", action="store_true",
                        help="Process all WSIs from cohort JSON")
    
    args = parser.parse_args()
    
    # Special excluded tiles (from original script)
    excluded_list = ['TV0577_68860_175674.png', 'TV0577_67836_167226.png']
    
    if args.process_all and args.cohort_json:
        # Batch processing mode
        with open(args.cohort_json, 'r') as f:
            cohort = json.load(f)
        
        cohort_with_tiles = deepcopy(cohort)
        
        for group in ['training', 'validation', 'test']:
            for label in ['ALL', 'AML_APL', 'CML', 'Lymphoma_CLL', 'MM']:
                cohort_with_tiles[group][label] = {'ids': cohort[group][label], 'tiles': []}
                
                for slide_id in cohort[group][label]:
                    print(f"\n{'='*60}")
                    print(f"Processing {slide_id} ({group}/{label})")
                    print(f"{'='*60}")
                    
                    excluded = excluded_list if slide_id == 'TV0577' else None
                    selected_tiles = process_single_wsi(
                        slide_id=slide_id,
                        label=label,
                        group=group,
                        wsi_source_dir=args.wsi_source_dir,
                        dest_parent_dir=args.dest_parent_dir,
                        num_tiles=args.num_tiles,
                        excluded_tiles=excluded
                    )
                    
                    cohort_with_tiles[group][label]['tiles'].extend(selected_tiles)
        
        # Save cohort with tile information
        output_json = Path(args.dest_parent_dir) / '1202_cohort_tiles_random.json'
        with open(output_json, 'w') as f:
            json.dump(cohort_with_tiles, f, indent=2)
        print(f"\nSaved cohort information to {output_json}")
        
    elif args.slide_id and args.label and args.group:
        # Single WSI processing mode
        print(f"Processing single WSI: {args.slide_id}")
        excluded = excluded_list if args.slide_id == 'TV0577' else None
        
        selected_tiles = process_single_wsi(
            slide_id=args.slide_id,
            label=args.label,
            group=args.group,
            wsi_source_dir=args.wsi_source_dir,
            dest_parent_dir=args.dest_parent_dir,
            num_tiles=args.num_tiles,
            excluded_tiles=excluded
        )
        
        print(f"\nSelected {len(selected_tiles)} tiles for {args.slide_id}")
    else:
        parser.print_help()
        print("\nError: Either provide --slide_id, --label, and --group for single WSI,")
        print("       or use --process_all with --cohort_json for batch processing")

if __name__ == "__main__":
    main()
