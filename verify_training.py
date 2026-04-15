"""
Training Verification Script

Check training status, verify models are saved, and monitor progress.

Usage:
    python verify_training.py                    # Check all training stages
    python verify_training.py --status           # Quick status check
    python verify_training.py --list-models      # List all saved models
    python verify_training.py --list-results     # List all results files
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model_manager import ModelManager


class TrainingVerifier:
    """Verify training completion and model persistence."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.models_dir = Path('models')
        self.results_dir = Path('results')
        self.checkpoints_dir = Path('checkpoints')
    
    def print_header(self, text: str):
        """Print formatted header."""
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70)
    
    def check_directories(self):
        """Check if all required directories exist."""
        self.print_header("DIRECTORY STATUS")
        
        dirs = {
            'Models': self.models_dir,
            'Results': self.results_dir,
            'Checkpoints': self.checkpoints_dir,
        }
        
        for name, path in dirs.items():
            status = "✓ EXISTS" if path.exists() else "✗ MISSING"
            size = f"({len(list(path.glob('*')))} items)" if path.exists() else ""
            print(f"{name:15} {status:15} {path.absolute()} {size}")
    
    def check_models(self) -> Dict:
        """Check saved models."""
        self.print_header("SAVED MODELS")
        
        models = list(self.models_dir.glob("*.pt")) if self.models_dir.exists() else []
        
        if not models:
            print("✗ NO MODELS FOUND")
            return {}
        
        print(f"✓ Found {len(models)} model(s):\n")
        
        model_info = {}
        for model_file in sorted(models):
            # Try to load metadata
            meta_file = model_file.with_suffix('') .name + '_metadata.json'
            meta_path = self.models_dir / meta_file
            
            info = {
                'file': model_file.name,
                'size_mb': round(model_file.stat().st_size / (1024 * 1024), 2),
                'created': datetime.fromtimestamp(model_file.stat().st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                        if 'timestamp' in meta:
                            info['timestamp'] = meta['timestamp']
                        if 'metrics' in meta:
                            info['metrics'] = meta['metrics']
                except Exception as e:
                    print(f"  Warning: Could not read metadata: {e}")
            
            model_info[model_file.name] = info
            
            print(f"  📦 {model_file.name}")
            print(f"     Size: {info['size_mb']} MB")
            print(f"     Created: {info['created']}")
            if 'metrics' in info and info['metrics']:
                for key, value in info['metrics'].items():
                    if isinstance(value, float):
                        print(f"     {key}: {value:.4f}")
                    else:
                        print(f"     {key}: {value}")
            print()
        
        return model_info
    
    def check_results(self) -> Dict:
        """Check saved results files."""
        self.print_header("SAVED RESULTS")
        
        results = list(self.results_dir.glob("*.json")) if self.results_dir.exists() else []
        
        if not results:
            print("✗ NO RESULTS FOUND")
            return {}
        
        print(f"✓ Found {len(results)} result file(s):\n")
        
        results_info = {}
        for result_file in sorted(results):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                
                info = {
                    'file': result_file.name,
                    'phase': data.get('phase', 'unknown'),
                    'timestamp': data.get('timestamp', 'unknown'),
                }
                
                # Extract key metrics
                if 'training_results' in data:
                    tr = data['training_results']
                    if 'test_accuracy' in tr:
                        info['test_accuracy'] = tr['test_accuracy']
                    if 'gate_passed' in tr:
                        info['gate_passed'] = tr['gate_passed']
                
                if 'evaluation_metrics' in data:
                    em = data['evaluation_metrics']
                    if 'accuracy' in em:
                        info['accuracy'] = em['accuracy']
                
                results_info[result_file.name] = info
                
                print(f"  📄 {result_file.name}")
                print(f"     Phase: {info.get('phase', 'N/A')}")
                print(f"     Timestamp: {info.get('timestamp', 'N/A')}")
                if 'accuracy' in info:
                    print(f"     Accuracy: {info['accuracy']:.4f}")
                if 'test_accuracy' in info:
                    print(f"     Test Accuracy: {info['test_accuracy']:.4f}")
                if 'gate_passed' in info:
                    print(f"     Gate Passed: {info['gate_passed']}")
                print()
            
            except Exception as e:
                print(f"  ✗ Error reading {result_file.name}: {e}\n")
        
        return results_info
    
    def check_training_completion(self) -> Dict:
        """Check if training phases completed."""
        self.print_header("TRAINING COMPLETION STATUS")
        
        phases = {
            'Phase A': {
                'model': 'phase_a_model.pt',
                'results': 'phase_a_results.json',
            },
            'Phase B': {
                'model': 'best_model.pt',
                'results': 'phase_b_results.json',
            },
            'Validation': {
                'model': 'validation_model.pt',
                'results': 'validation_results.json',
            },
        }
        
        completion = {}
        for phase, files in phases.items():
            model_exists = (self.models_dir / files['model']).exists()
            results_exists = (self.results_dir / files['results']).exists()
            
            status = "✓ COMPLETE" if (model_exists and results_exists) else "✗ INCOMPLETE"
            completion[phase] = {
                'model': model_exists,
                'results': results_exists,
                'status': status,
            }
            
            print(f"{phase:15} {status}")
            print(f"  Model:   {'✓' if model_exists else '✗'} {files['model']}")
            print(f"  Results: {'✓' if results_exists else '✗'} {files['results']}")
            print()
        
        return completion
    
    def get_summary(self) -> Dict:
        """Get comprehensive training summary."""
        self.print_header("TRAINING SUMMARY")
        
        models = list(self.models_dir.glob("*.pt")) if self.models_dir.exists() else []
        results = list(self.results_dir.glob("*.json")) if self.results_dir.exists() else []
        
        summary = {
            'total_models': len(models),
            'total_results': len(results),
            'best_model_exists': (self.models_dir / 'best_model.pt').exists(),
            'all_files': {
                'models': [f.name for f in sorted(models)],
                'results': [f.name for f in sorted(results)],
            }
        }
        
        print(f"Total Models Saved: {len(models)}")
        print(f"Total Results Files: {len(results)}")
        print(f"Best Model Available: {'✓ YES' if summary['best_model_exists'] else '✗ NO'}")
        
        if models:
            print(f"\nModels: {', '.join([m.name for m in sorted(models)])}")
        if results:
            print(f"\nResults: {', '.join([r.name for r in sorted(results)])}")
        
        return summary
    
    def verify_all(self):
        """Run all verifications."""
        self.check_directories()
        self.check_models()
        self.check_results()
        self.check_training_completion()
        self.get_summary()
        
        self.print_header("VERIFICATION COMPLETE")


def main():
    parser = argparse.ArgumentParser(description='Verify training completion and model persistence')
    parser.add_argument('--status', action='store_true', help='Show quick status')
    parser.add_argument('--list-models', action='store_true', help='List all models')
    parser.add_argument('--list-results', action='store_true', help='List all results')
    parser.add_argument('--summary', action='store_true', help='Show summary')
    parser.add_argument('--all', action='store_true', help='Run all checks (default)')
    
    args = parser.parse_args()
    verifier = TrainingVerifier()
    
    if args.status:
        verifier.check_training_completion()
    elif args.list_models:
        verifier.check_models()
    elif args.list_results:
        verifier.check_results()
    elif args.summary:
        verifier.get_summary()
    else:
        # Default: run all verifications
        verifier.verify_all()


if __name__ == '__main__':
    main()
