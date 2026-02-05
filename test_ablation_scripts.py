"""
Quick verification test for ablation study scripts.
Tests basic functionality without requiring full dataset.
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil
import json
import numpy as np
from PIL import Image

def create_test_tiles(test_dir, slide_id, num_tiles=250):
    """Create dummy tile images for testing."""
    tile_folder = test_dir / f"{slide_id}_tiles"
    tile_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_tiles} test tiles in {tile_folder}")
    
    for i in range(num_tiles):
        # Create a random image
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save with naming convention: SLIDE_X_Y.png
        tile_path = tile_folder / f"{slide_id}_{i*1000}_{i*2000}.png"
        img.save(tile_path)
    
    print(f"Created {num_tiles} tiles")
    return tile_folder

def create_test_cohort(cohort_path):
    """Create a minimal test cohort JSON."""
    cohort = {
        "training": {
            "ALL": ["TEST01", "TEST02"],
            "AML_APL": ["TEST03"],
            "CML": [],
            "Lymphoma_CLL": [],
            "MM": []
        },
        "validation": {
            "ALL": [],
            "AML_APL": [],
            "CML": [],
            "Lymphoma_CLL": [],
            "MM": []
        },
        "test": {
            "ALL": ["TEST04"],
            "AML_APL": [],
            "CML": [],
            "Lymphoma_CLL": [],
            "MM": []
        }
    }
    
    with open(cohort_path, 'w') as f:
        json.dump(cohort, f, indent=2)
    
    print(f"Created test cohort at {cohort_path}")
    return cohort

def test_random_selection():
    """Test the random tile selection script."""
    print("\n" + "="*60)
    print("TEST: Random Tile Selection")
    print("="*60)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tile_source = temp_path / "tiles"
        dest_dir = temp_path / "selected"
        cohort_path = temp_path / "test_cohort.json"
        
        # Create test data
        cohort = create_test_cohort(cohort_path)
        
        # Create tiles for each slide in cohort
        for group in cohort:
            for label in cohort[group]:
                for slide_id in cohort[group][label]:
                    create_test_tiles(tile_source, slide_id, num_tiles=250)
        
        # Test single WSI processing
        print("\n--- Testing Single WSI Processing ---")
        sys.path.insert(0, str(Path(__file__).parent))
        from importlib import import_module
        
        # Import the module
        random_roi_module = import_module('4_choose_200_random_ROIs_single_wsi')
        
        selected_tiles = random_roi_module.process_single_wsi(
            slide_id="TEST01",
            label="ALL",
            group="training",
            tile_source_dir=str(tile_source),
            dest_parent_dir=str(dest_dir),
            num_tiles=200,
            cleanup=False  # Don't cleanup for testing
        )
        
        # Verify results
        assert len(selected_tiles) == 200, f"Expected 200 tiles, got {len(selected_tiles)}"
        
        dest_tiles = list((dest_dir / "training" / "ALL").glob("*.png"))
        assert len(dest_tiles) == 200, f"Expected 200 files in destination, got {len(dest_tiles)}"
        
        print(f"✓ Single WSI test passed: {len(selected_tiles)} tiles selected")
        
        # Test batch processing
        print("\n--- Testing Batch Processing ---")
        dest_dir2 = temp_path / "selected_batch"
        
        # Run batch processing via command line
        import subprocess
        cmd = [
            sys.executable,
            "4_choose_200_random_ROIs_single_wsi.py",
            "--process_all",
            "--cohort_json", str(cohort_path),
            "--tile_source_dir", str(tile_source),
            "--dest_parent_dir", str(dest_dir2),
            "--num_tiles", "200",
            "--no_cleanup"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✓ Batch processing completed successfully")
            
            # Verify output JSON exists
            output_json = dest_dir2 / "1202_cohort_tiles_random.json"
            assert output_json.exists(), "Output JSON not created"
            print(f"✓ Output JSON created: {output_json}")
            
            # Verify tiles were copied
            total_tiles = sum(1 for _ in dest_dir2.rglob("*.png"))
            expected_tiles = 200 * 4  # 4 slides with 200 tiles each
            assert total_tiles == expected_tiles, f"Expected {expected_tiles} total tiles, got {total_tiles}"
            print(f"✓ All tiles copied: {total_tiles} tiles")
        else:
            print(f"✗ Batch processing failed:")
            print(result.stderr)
            return False
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    return True

def main():
    """Run verification tests."""
    print("\n" + "#"*60)
    print("# Ablation Study Scripts Verification")
    print("#"*60)
    
    try:
        success = test_random_selection()
        
        if success:
            print("\n✓ Verification complete! Scripts are ready to use.")
            print("\nNext steps:")
            print("1. Run random tile selection on your actual dataset")
            print("2. Train a model using New_Classification.py")
            print("3. Test the model using New_Testing.py")
            print("4. Compare results using compare_ablation_results.py")
            print("\nSee ABLATION_STUDY_README.md for detailed usage instructions.")
        else:
            print("\n✗ Verification failed. Please check the error messages above.")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
