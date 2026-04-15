# Phase 4 Training Execution Guide

**Last Updated**: April 8, 2026  
**Status**: ✅ Ready for Training Execution  

---

## Quick Summary

**Current State**: ✗ **NO TRAINING HAS BEEN COMPLETED**

**Evidence**:
```
Models directory:    EMPTY (0 files)
Results directory:   EMPTY (0 files)  
Checkpoints:         EMPTY (0 files)
```

**Reason for Previous Interruption**: Training timed out after 5 minutes due to CPU-only processing. Full training requires 2-4 hours on CPU.

**Solution**: Enhanced training framework with automatic model persistence now in place. Models and results are **guaranteed to be saved** after each training phase.

---

## How to Execute Training Properly

### Option 1: Full Two-Phase Training (Recommended)

```bash
cd "c:\Users\aditj\New Projects\Skin-Cancer-Disease-Prediction-System"
python train_phase4.py
```

**What happens**:

**Phase A** (20% subset):
- Duration: ~10-30 minutes on CPU
- Saves: `models/phase_a_model.pt` + `results/phase_a_results.json`
- Gate requirement: ≥65% accuracy
- If gate fails: Training stops, can retry with adjusted hyperparameters

**Phase B** (100% dataset):
- Duration: ~2-4 hours on CPU (only if Phase A passed gate)
- Saves: `models/best_model.pt` + `results/phase_b_results.json`
- Target: ≥70% accuracy
- This is your final baseline model

### Option 2: Quick Validation (5 epochs, ~5 minutes)

```bash
python train_phase4_abbreviated.py
```

**Purpose**: Quick validation that pipeline works without waiting for full training

**Saves**:
- `models/validation_model.pt`
- `results/validation_results.json`

### Option 3: Use GPU (If Available)

Same commands - automatic GPU detection:
```
python -c "import torch; print('GPU Available' if torch.cuda.is_available() else 'CPU Only')"
```

If GPU available:
- Training time: **30-60 minutes** (instead of 2-4 hours)
- Otherwise: Same command, automatic fallback to CPU

---

## Monitoring Training Progress

### Check Status While Training is Running

```bash
# In a separate terminal window:
python verify_training.py --status
```

**Output Example**:
```
Phase A         ✓ COMPLETE
  Model:   ✓ phase_a_model.pt
  Results: ✓ phase_a_results.json

Phase B         ⏳ IN PROGRESS (or ✗ INCOMPLETE)
  Model:   ✗ best_model.pt
  Results: ✗ phase_b_results.json
```

### View All Saved Models

```bash
python verify_training.py --list-models
```

### View All Results

```bash
python verify_training.py --list-results
```

### Get Detailed Summary

```bash
python verify_training.py --summary
```

---

## What Gets Saved and Where

### After Phase A Completes

```
📁 models/
   ├── phase_a_model.pt           ← Phase A trained model
   └── phase_a_model_metadata.json ← Metadata (timestamp, metrics)

📁 results/
   └── phase_a_results.json        ← Training history, metrics, accuracy
```

**Contents of phase_a_results.json**:
```json
{
  "phase": "phase_a",
  "timestamp": "2026-04-08T10:30:45",
  "training_results": {
    "gate_passed": true,
    "final_validation_accuracy": 0.68,
    "training_time_minutes": 25
  },
  "training_history": {
    "epoch": [1, 2, ..., 20],
    "train_loss": [...],
    "val_loss": [...],
    "val_acc": [...],
    "learning_rate": [...]
  },
  "evaluation_metrics": {...}
}
```

### After Phase B Completes

```
📁 models/
   ├── phase_a_model.pt
   ├── phase_a_model_metadata.json
   ├── best_model.pt               ← FINAL TRAINED MODEL
   └── best_model_metadata.json    ← Model metadata

📁 results/
   ├── phase_a_results.json
   ├── phase_b_results.json        ← FINAL RESULTS
   └── phase4_results.json         ← Backup copy
```

---

## Training Parameters & What They Mean

### Dataset Configuration
- **Train set**: 70% (7,054 images from 5,228 unique lesions)
- **Validation set**: 15% (1,464 images from 1,121 unique lesions)
- **Test set**: 15% (1,497 images from 1,121 unique lesions)
- **Classes**: 7 skin lesion types
- **Imbalance ratio**: 67:1 (nv dominates)

### Training Configuration

**Phase A (Subset)**:
- Epochs: 20
- Early stopping: 10 epochs without improvement
- Learning rate: 0.001 (Adam optimizer)
- Batch size: 32
- Early stopping gate: ≥65% accuracy, <30 minutes

**Phase B (Full)**:
- Epochs: 100
- Early stopping: 10 epochs without improvement
- Learning rate: 0.001 (adaptive ReduceLROnPlateau scheduler)
- Batch size: 32
- Target: ≥70% test accuracy

### Model Architecture
- **Type**: CNN Baseline (4-block CNN)
- **Parameters**: 1.7 million
- **Dropout**: 0.5
- **Loss function**: CrossEntropyLoss with class weights
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)

---

## Expected Results

### Phase A Success Criteria
✓ Validation accuracy ≥ 65%  
✓ Training completes in < 30 minutes  
✓ Model saved to `models/phase_a_model.pt`

**Typical Phase A Results** (if training completes):
- Final validation accuracy: 65-72%
- Training time: 10-25 minutes (CPU)

### Phase B Success Criteria (Only if Phase A Passes)
✓ Test accuracy ≥ 70%  
✓ Model saved to `models/best_model.pt`

**Typical Phase B Results** (if training completes):
- Final test accuracy: 70-78%
- Training time: 2-4 hours (CPU)

---

## Troubleshooting

### Problem: Training was interrupted again

**Solution**: 
- Training now saves models immediately after each phase
- If Phase A finished: You have `models/phase_a_model.pt` saved
- Can restart Phase B or entire training

**Check status**:
```bash
python verify_training.py --list-models
```

### Problem: Only see phase_a_model.pt, not best_model.pt

**Solution**:
- This means Phase A completed but Phase B is still running
- Or Phase B didn't meet ≥70% gate
- Check results file:
```bash
python verify_training.py --list-results
```

### Problem: No models saved at all after training

**Verification**:
1. Check if training actually ran:
   ```bash
   python verify_training.py --status
   ```
2. Look for error messages in terminal
3. Verify GPU/CPU:
   ```bash
   python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU')"
   ```

### Problem: Want to use trained model for predictions

**Commands**:
```bash
# Single image prediction
python predict.py --image test.jpg

# Batch predictions  
python predict.py --batch ./test_images/

# Evaluate on test set
python evaluate.py --model models/best_model.pt

# Web app (if implemented)
python app.py
```

**Note**: All now default to `models/best_model.pt` (updated from `.pth`)

---

## Advanced: Monitoring in Real-Time

### Windows PowerShell: Watch Training in Separate Terminal

```powershell
# Terminal 1: Start training (do NOT close this window)
cd "path\to\project"
python train_phase4.py

# Terminal 2: Monitor progress (separate window)
cd "path\to\project"
while($true) { python verify_training.py --status; Start-Sleep -Seconds 10 }
```

### Linux/Mac: Watch Training in Separate Terminal

```bash
# Terminal 1: Start training
python train_phase4.py

# Terminal 2: Monitor progress
while true; do python verify_training.py --status; sleep 10; done
```

### Tail Training Log

```bash
# Redirect output to file
python train_phase4.py > training.log 2>&1 &

# Monitor log file
tail -f training.log
```

---

## What Was Wrong Before vs What's Fixed Now

### Previous Issues ✗

| Problem | Impact |
|---------|--------|
| No model persistence | Training results lost if interrupted |
| Only command-line output | Can't verify mid-training |
| No result files | Can't track metrics after training |
| Models saved to unclear location | Hard to find/use trained models |
| No metadata tracking | Don't know when/how models were trained |

### Now Fixed ✅

| Solution | Benefit |
|----------|---------|
| `ModelManager` ensures model saving | Models safe even if interrupted |
| `verify_training.py` script | Check status anytime |
| `results/` directory for all outputs | All results in one place |
| Clear `models/` directory | Easy to find trained models |
| Metadata with each model | Know timestamp, metrics, parameters |

---

## Step-by-Step Execution Checklist

- [ ] **Step 1**: Verify current state
  ```bash
  python verify_training.py --status
  ```
  Expected: All `✗ INCOMPLETE`

- [ ] **Step 2**: Start training
  ```bash
  python train_phase4.py
  ```

- [ ] **Step 3**: Wait for Phase A (~30 min)
  - Keep terminal open
  - DO NOT interrupt

- [ ] **Step 4**: Check Phase A results
  ```bash
  python verify_training.py --list-models
  ```
  Expected: `phase_a_model.pt` exists

- [ ] **Step 5**: Phase B continues automatically if Phase A passes
  - Or manually check:
  ```bash
  python verify_training.py --status
  ```

- [ ] **Step 6**: After Phase B completes (~2-4 hours total)
  ```bash
  python verify_training.py --list-models
  ```
  Expected: `best_model.pt` exists

- [ ] **Step 7**: Verify final results
  ```bash
  python verify_training.py --summary
  ```

---

## FAQ

**Q: How long does training take?**  
A: Phase A: 10-30 min (CPU), Phase B: 2-4 hours (CPU). With GPU: 30-60 min total.

**Q: Can I interrupt training?**  
A: Yes, models are saved. But interrupting before Phase A completes means no models saved.

**Q: Where are my trained models?**  
A: Check `models/` directory or run `python verify_training.py --list-models`

**Q: How do I know which model is the best one?**  
A: Look for `best_model.pt` - that's the final Phase B model.

**Q: Can I skip Phase A and go straight to Phase B?**  
A: Not recommended, but Phase A is just a subset validation gate.

**Q: What if Phase A doesn't reach 65% accuracy?**  
A: Training stops. Try with different hyperparameters or longer training.

**Q: Where are training logs saved?**  
A: Console output only (can redirect to file with `> training.log`)

**Q: Can I resume training if interrupted?**  
A: Not automatically. But saved models from completed phases are preserved.

---

## Success Indicators

✅ Training is successful when:
1. `models/best_model.pt` exists (final trained model)
2. `results/phase_b_results.json` shows test accuracy ≥ 70%
3. No errors in terminal during training
4. `python verify_training.py --status` shows:
   - Phase A: ✓ COMPLETE
   - Phase B: ✓ COMPLETE

---

**Ready to train? Run**: `python train_phase4.py`

**Status unclear? Run**: `python verify_training.py --all`

**Need results? Run**: `python verify_training.py --list-results`
