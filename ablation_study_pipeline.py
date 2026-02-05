"""
Ablation Study Pipeline Orchestrator
Coordinates the entire ablation study workflow:
1. Random tile selection
2. Model training (optional)
3. Model testing
4. Results comparison
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

def run_command(cmd, description, check=True):
    """
    Execute a shell command and handle errors.
    
    Args:
        cmd: Command to execute (list or string)
        description: Description of the command for logging
        check: Whether to raise exception on failure
    
    Returns:
        CompletedProcess object
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        # Stream output in real-time instead of capturing it
        result = subprocess.run(cmd, check=check)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if check:
            raise
        return e

def random_tile_selection(cohort_json, wsi_source_dir, dest_parent_dir, num_tiles=200):
    """
    Step 1: Random tile selection for all WSIs in cohort.
    Tiles WSI files directly and saves only randomly selected tiles.
    """
    cmd = [
        sys.executable,
        "4_choose_200_random_ROIs_single_wsi.py",
        "--process_all",
        "--cohort_json", cohort_json,
        "--wsi_source_dir", wsi_source_dir,
        "--dest_parent_dir", dest_parent_dir,
        "--num_tiles", str(num_tiles)
    ]
    
    run_command(cmd, "Step 1: Random Tile Selection (Tiling + Selection)")

def train_classification_model(dataset_dir, exp_name, epochs=100, freeze_epochs=None, batch_size=8, learning_rate=0.001):
    """
    Step 2: Train disease classification model on randomly selected tiles.
    Uses simplified Ablation_Classification.py (DenseNet121 only, no SLURM).
    """
    # Split epochs into freeze and fine-tune (10% freeze, 90% fine-tune)
    freeze_epochs = freeze_epochs if isinstance(freeze_epochs, int) else max(1, int(epochs * 0.1))
    fine_tune_epochs = epochs - freeze_epochs
    
    cmd = [
        sys.executable,
        "Ablation_Classification.py",
        dataset_dir,
        exp_name,
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--freeze_epochs", str(freeze_epochs),
        "--fine_tune_epochs", str(fine_tune_epochs)
    ]
    
    run_command(cmd, "Step 2: Training Classification Model (DenseNet121)")

def test_model(dataset_dir, exp_dir, model_path=None, save_dir=None, batch_size=8):
    """
    Step 3: Test the trained model and generate metrics.
    """
    cmd = [
        sys.executable,
        "New_Testing.py",
        dataset_dir,
        exp_dir,
        "--batch_size", str(batch_size)
    ]
    
    if model_path:
        cmd.extend(["--model_path", model_path])
    if save_dir:
        cmd.extend(["--save_dir", save_dir])
    
    run_command(cmd, "Step 3: Testing Model")

def compare_results(quality_based_dir, random_based_dir, output_dir):
    """
    Step 4: Compare results between quality-based and random selection.
    """
    cmd = [
        sys.executable,
        "compare_ablation_results.py",
        "--quality_dir", quality_based_dir,
        "--random_dir", random_based_dir,
        "--output_dir", output_dir
    ]
    
    run_command(cmd, "Step 4: Comparing Results", check=False)

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate ablation study pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Step selection
    parser.add_argument("--step", choices=["all", "select", "train", "test", "compare"],
                       default="all", help="Which step(s) to run")
    
    # Tile selection arguments
    parser.add_argument("--cohort_json", default="../Final_BM_cytology_all_200/1202_cohort.json",
                       help="Path to cohort JSON file")
    parser.add_argument("--wsi_source_dir", default="../Bone_marrow_cytology_WSI",
                       help="Source directory containing .mrxs WSI files")
    parser.add_argument("--random_dest_dir", default="../Final_BM_cytology_all_200_random",
                       help="Destination directory for randomly selected tiles")
    parser.add_argument("--num_tiles", type=int, default=200,
                       help="Number of tiles to randomly select per WSI")
    
    # Training arguments
    parser.add_argument("--train_dataset", default="../Final_BM_cytology_all_200_random",
                       help="Dataset directory for training")
    parser.add_argument("--exp_name", default=None,
                       help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Total number of training epochs")
    parser.add_argument("--freeze_epochs", default=None,
                       help="Number of freezing epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training/testing")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Initial learning rate")
    
    # Testing arguments
    parser.add_argument("--test_dataset", default="../Final_BM_cytology_all_200_random/test",
                       help="Test dataset directory")
    parser.add_argument("--exp_dir", default=None,
                       help="Experiment directory containing trained model")
    parser.add_argument("--model_path", default=None,
                       help="Specific model path to test")
    parser.add_argument("--test_save_dir", default=None,
                       help="Directory to save test results")
    
    # Comparison arguments
    parser.add_argument("--quality_results_dir", default=None,
                       help="Directory with quality-based selection results")
    parser.add_argument("--random_results_dir", default=None,
                       help="Directory with random selection results")
    parser.add_argument("--comparison_output_dir", default="./ablation_comparison",
                       help="Directory to save comparison results")
    
    args = parser.parse_args()
    
    # Auto-generate experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"ablation_random_densenet121_{timestamp}"
    
    print(f"\n{'#'*60}")
    print(f"# Ablation Study Pipeline")
    print(f"# Experiment: {args.exp_name}")
    print(f"{'#'*60}\n")
    
    try:
        # Step 1: Random tile selection (tiles WSI directly)
        if args.step in ["all", "select"]:
            random_tile_selection(
                cohort_json=args.cohort_json,
                wsi_source_dir=args.wsi_source_dir,
                dest_parent_dir=args.random_dest_dir,
                num_tiles=args.num_tiles
            )
        
        # Step 2: Train model
        if args.step in ["all", "train"]:
            train_classification_model(
                dataset_dir=args.train_dataset,
                exp_name=args.exp_name,
                epochs=args.epochs,
                freeze_epochs=args.freeze_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
        
        # Step 3: Test model
        if args.step in ["all", "test"]:
            exp_dir = args.exp_dir if args.exp_dir else f"../BM_cytology_classification_logs/{args.exp_name}"
            test_model(
                dataset_dir=args.test_dataset,
                exp_dir=exp_dir,
                model_path=args.model_path,
                save_dir=args.test_save_dir,
                batch_size=args.batch_size
            )
        
        # Step 4: Compare results
        if args.step in ["all", "compare"]:
            if args.quality_results_dir and args.random_results_dir:
                compare_results(
                    quality_based_dir=args.quality_results_dir,
                    random_based_dir=args.random_results_dir,
                    output_dir=args.comparison_output_dir
                )
            else:
                print("\nSkipping comparison step: --quality_results_dir and --random_results_dir required")
        
        print(f"\n{'#'*60}")
        print(f"# Pipeline completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"! Pipeline failed with error:")
        print(f"! {e}")
        print(f"{'!'*60}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
