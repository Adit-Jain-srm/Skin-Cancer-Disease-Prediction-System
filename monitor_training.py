#!/usr/bin/env python
"""Real-time training progress monitor."""

import time
import subprocess
import sys
from pathlib import Path

def get_latest_checkpoint():
    """Get the latest checkpoint info."""
    checkpoint_dir = Path('checkpoints')
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob('*.pt'))
    if not checkpoints:
        return None
    
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    size_mb = latest.stat().st_size / (1024 * 1024)
    return {
        'name': latest.name,
        'size_mb': size_mb,
        'mtime': latest.stat().st_mtime
    }

def check_saved_models():
    """Check saved models and results."""
    models_dir = Path('models')
    results_dir = Path('results')
    
    models = {}
    if models_dir.exists():
        for model in models_dir.glob('*.pt'):
            models[model.name] = model.stat().st_size / (1024 * 1024)
    
    results = {}
    if results_dir.exists():
        for result in results_dir.glob('*.json'):
            results[result.name] = result.stat().st_size / 1024
    
    return models, results

def monitor():
    """Monitor training progress."""
    print("=" * 70)
    print("TRAINING MONITOR")
    print("=" * 70)
    
    last_checkpoint = None
    update_count = 0
    
    try:
        while True:
            update_count += 1
            
            # Check checkpoints
            checkpoint = get_latest_checkpoint()
            checkpoint_changed = checkpoint != last_checkpoint if last_checkpoint else False
            
            # Check saved models and results
            models, results = check_saved_models()
            
            # Clear and print status
            if update_count > 1:
                print("\033[H\033[J", end="")  # Clear screen (Unix-like)
            
            print(f"\n[{time.strftime('%H:%M:%S')}] Update #{update_count}")
            print("-" * 70)
            
            # Latest checkpoint
            if checkpoint:
                print(f"✓ Latest checkpoint: {checkpoint['name']}")
                print(f"  Size: {checkpoint['size_mb']:.1f} MB")
                if checkpoint_changed:
                    print("  ★ UPDATED")
            else:
                print("⏳ No checkpoint yet (training still loading data...)")
            
            print("\nSaved Models:")
            if models:
                for name, size_mb in models.items():
                    print(f"  ✓ {name} ({size_mb:.1f} MB)")
                    if name == 'phase_a_model.pt':
                        print("    → Phase A COMPLETED")
                    elif name == 'best_model.pt':
                        print("    → Phase B COMPLETED / BEST MODEL")
            else:
                print("  (none yet)")
            
            print("\nResults:")
            if results:
                for name, size_kb in results.items():
                    print(f"  ✓ {name} ({size_kb:.1f} KB)")
                    if 'phase_a' in name:
                        print("    → Phase A metrics saved")
                    elif 'phase_b' in name:
                        print("    → Phase B metrics saved")
            else:
                print("  (none yet)")
            
            # Status summary
            print("\n" + "-" * 70)
            phase_a_done = 'phase_a_model.pt' in models and 'phase_a_results.json' in results
            phase_b_done = 'best_model.pt' in models and 'phase_b_results.json' in results
            
            if phase_a_done and phase_b_done:
                print("✅ TRAINING COMPLETE - All phases finished!")
                break
            elif phase_a_done:
                print("⏳ Phase A DONE, Phase B in progress...")
            elif checkpoint:
                print("⏳ Phase A training in progress...")
            else:
                print("⏳ Initializing training...")
            
            last_checkpoint = checkpoint
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\n[Monitor stopped by user]")
        sys.exit(0)

if __name__ == '__main__':
    monitor()
