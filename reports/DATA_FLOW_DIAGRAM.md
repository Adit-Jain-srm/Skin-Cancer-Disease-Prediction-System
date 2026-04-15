# Data Flow Diagrams (DFD)
## Skin Cancer Disease Prediction System

**Phase**: 2 (Analysis & Design)  
**Date**: 2026-04-08  
**Version**: 2.0 (Enhanced from Phase 1)  
**Status**: ✅ **APPROVED**

---

## 1. Context Diagram (Level 0)

```
                    ┌─────────────┐
                    │    END USER │
                    │  (Browser)  │
                    └──────┬──────┘
                           │
                    Image Upload
                    Prediction Request
                           │
                           ↓
            ┌──────────────────────────────────┐
            │  SKIN CANCER PREDICTION SYSTEM   │
            │                                  │
            │  • Upload & validation           │
            │  • Preprocessing                 │
            │  • Model inference               │
            │  • Result display                │
            └──────────────────────────────────┘
                           │
                    Prediction Result
                    Class + Confidence
                           │
                           ↓
                    ┌─────────────┐
                    │    USER     │
                    │  (Display)  │
                    └─────────────┘


                    ┌──────────────┐
                    │  DATASET    │
                    │  Directory  │
                    │ (Images +   │
                    │  Metadata)  │
                    └──────┬───────┘
                           │
                    Load metadata
                    Load images
                           │
                           ↓
            ┌──────────────────────────────────┐
            │  SKIN CANCER PREDICTION SYSTEM   │
            └──────────────────────────────────┘


                    ┌──────────────┐
                    │ MODEL        │
                    │ CHECKPOINT   │
                    │ (weights.pth)│
                    └──────┬───────┘
                           │
                    Load weights
                           │
                           ↓
            ┌──────────────────────────────────┐
            │  SKIN CANCER PREDICTION SYSTEM   │
            └──────────────────────────────────┘
```

---

## 2. Level 1 DFD: Main Prediction Flow

```
                          USER
                           │
                           ↓
                    ┌─────────────┐
                    │  Upload File│
     ┌──────────────│   (Web UI)  │─────────────┐
     │              └─────────────┘             │
     │ File uploaded                            │ Form submitted
     ↓                                          ↓
 ┌────────────────────┐              ┌──────────────────────┐
 │ PROCESS 1          │              │ PROCESS 2            │
 │ FILE VALIDATION    │              │ REQUEST PARSING      │
 │                    │              │                      │
 │ • Check extension  │              │ • Extract form data  │
 │ • Check file size  │              │ • Get file object    │
 │ • Verify MIME type │              │ • Log request        │
 │                    │              │                      │
 │ Valid: YES/NO      │              │ Parsed request       │
 └────┬───────────────┘              └──────┬───────────────┘
      │                                     │
      ├─NO─→ ERROR STORE → Send error response
      │
      YES
      │
      ↓
 ┌────────────────────┐
 │ PROCESS 3          │
 │ IMAGE PREPROCESSING│
 │                    │
 │ @ DatasetManager   │
 │ • Load as PIL      │
 │ • Resize 224×224   │
 │ • Normalize [0,1]  │
 │ • Convert to tensor│
 │                    │
 │ Preprocessed image │
 └────┬───────────────┘
      │
      ↓
 ┌────────────────────┐
 │ PROCESS 4          │
 │ MODEL INFERENCE    │
 │                    │
 │ @ CNNModel         │
 │ • Load weights     │
 │ • Forward pass     │
 │ • Get logits       │
 │ • Apply softmax    │
 │                    │
 │ Predictions: {}    │
 └────┬───────────────┘
      │
      ↓
 ┌────────────────────┐
 │ PROCESS 5          │
 │ RESULT FORMATTING  │
 │                    │
 │ • Create JSON      │
 │ • Add confidence   │
 │ • Add timestamp    │
 │ • Add metadata     │
 │                    │
 │ Result JSON        │
 └────┬───────────────┘
      │
      ↓
 ┌────────────────────┐
 │ SEND RESPONSE      │
 │ HTTP 200 + JSON    │
 └────┬───────────────┘
      │
      ↓
     USER SEES RESULT
```

---

## 3. Detailed Swimlane Diagram: Prediction Request

```
┌──────────────┬──────────────┬──────────────┬──────────────┬─────────────┐
│   CLIENT     │   FLASK APP  │   DATA LAYER │ BUSINESS LOG │  EXTERNAL   │
│  (Browser)   │  (app.py)    │ (dataset.py) │ (model.py)   │ (PyTorch)   │
├──────────────┼──────────────┼──────────────┼──────────────┼─────────────┤
│              │              │              │              │             │
│ User clicks  │              │              │              │             │
│ "Upload File"│              │              │              │             │
│    │         │              │              │              │             │
│    ├─POST /predict ─────────→              │              │             │
│    │         │              │              │              │             │
│    │         │ Receive file │              │              │             │
│    │         │    │         │              │              │             │
│    │         │    ├─ Check extension      │              │             │
│    │         │    ├─ Check size < 10MB   │              │             │
│    │         │    ├─ MIME type = image/? │              │             │
│    │         │    │         │              │              │             │
│    │         │    ├─ VALID?              │              │             │
│    │         │    │  │                    │              │             │
│    │         │    │  YES                  │              │             │
│    │         │    │   └────────────────────→              │             │
│    │         │    │         │ Load image  │              │             │
│    │         │    │         │ Resize: (600,450)→(224,224)│             │
│    │         │    │         │ Normalize: μ/σ             │             │
│    │         │    │         │ → numpy [0,1]             │             │
│    │         │    │         │────────────────────────────→             │
│    │         │    │         │              │ Load model weights      │
│    │         │    │         │              │ Forward pass (1×3×224×224)
│    │         │    │         │              │ Logits output           │
│    │         │    │         │←─────────────┤ Softmax probabilities   │
│    │         │    │←────────┴──────────────┤              │          │
│    │         │    │converted probs         │              │             │
│    │         │    │         │              │              │             │
│    │         │ Format JSON response        │              │             │
│    │         │ {                           │              │             │
│    │         │   "class": "melanoma",     │              │             │
│    │         │   "confidence": 0.87,     │              │             │
│    │         │   "timestamp": "...",     │              │             │
│    │         │   "processing_time_ms": 1250             │             │
│    │         │ }                          │              │             │
│    │←─HTTP 200 + JSON────────┤             │              │             │
│    │ Display result          │             │              │             │
│    │ • Class name            │             │              │             │
│    │ • Confidence %          │             │              │             │
│    │ • Processing time       │             │              │             │
│    │                         │             │              │             │
└────┴───────────┴─────────────┴─────────────┴──────────────┴─────────────┘

                              TIMING BREAKDOWN
    Upload: 10-100ms | Validation: 100ms | Preprocessing: 300-500ms
    Inference: 1000-4000ms | Formatting: 50ms | Network: 100ms
    ────────────────────────────────────────────────────────────
                    TOTAL: 1.5s - 4.7s (SLA: ≤5s)
```

---

## 4. Linear Process Flow with Error Handling

```
START: User submits form
  │
  ├─→ [ 1. VALIDATE FILE ]
  │   ├─ Extension: JPG/PNG?
  │   │  └─NO → ERROR: "Invalid format"  ─┐
  │   │                                   │
  │   ├─ Size: < 10MB?                    │
  │   │  └─NO → ERROR: "File too large"  ─┤
  │   │                                   │
  │   ├─ MIME: image/*?                   │
  │   │  └─NO → ERROR: "Not an image"    ─┤
  │   │                                   │
  │   └─ ALL PASS                         │
  │       │                               │
  ├─→ [ 2. LOAD IMAGE ]                   │
  │   ├─ Read from disk                   │
  │   │  └─FAIL → ERROR: "IO error"      ─┤
  │   │                                   │
  │   ├─ Parse PIL Image                  │
  │   │  └─FAIL → ERROR: "Corrupted"     ─┤
  │   │                                   │
  │   └─ SUCCESS                          │
  │       │                               │
  ├─→ [ 3. RESIZE & NORMALIZE ]           │
  │   ├─ PIL.resize(224, 224)             │
  │   ├─ Convert to numpy array           │
  │   ├─ Divide by 255 → [0,1]           │
  │   │  └─FAIL → ERROR: "Preprocessing"─┤
  │   │                                   │
  │   └─ tensor(1,3,224,224)              │
  │       │                               │
  ├─→ [ 4. LOAD MODEL ]                   │
  │   ├─ Check weights file exists        │
  │   │  └─NO → ERROR: "Model missing"   ─┤
  │   │                                   │
  │   ├─ Load weights                     │
  │   │  └─FAIL → ERROR: "Model corrupt"─┤
  │   │                                   │
  │   ├─ Move to device (CPU/GPU)         │
  │   │  └─FAIL → ERROR: "Device error"  ─┤
  │   │                                   │
  │   └─ model.eval()                     │
  │       │                               │
  ├─→ [ 5. INFERENCE ]                    │
  │   ├─ with torch.no_grad():            │
  │   │  ├─ output = model(tensor)        │
  │   │  │  └─FAIL → ERROR: "Inference"  ─┤
  │   │  ├─ probs = softmax(output)       │
  │   │  ├─ class_id = argmax(probs)      │
  │   │  └─ confidence = max(probs)       │
  │   │                                   │
  │   └─ results dict                     │
  │       │                               │
  ├─→ [ 6. FORMAT RESPONSE ]              │
  │   ├─ Create JSON:                     │
  │   │  ├─ class_name: CLASS_NAMES[id]   │
  │   │  ├─ confidence: round(conf, 2)    │
  │   │  ├─ timestamp: now()              │
  │   │  └─ processing_ms: time_taken     │
  │   │                                   │
  │   └─ JSON string                      │
  │       │                               │
  └─→ [ 7. SEND RESPONSE ]                │
      ├─ HTTP 200 OK ──────────────────────┘
      │   └─ Client sees result
      │
      └─ HTTP 400/500 ERROR ──────────────→
          (Errors from above)
          └─ Display user message


DATA STORES:
• Models/best_model.pth ← Model reads weights  
• /tmp/ ← Uploaded file temporarily stored
• config.yaml ← Model config read
• logs/error.log ← Errors logged
```

---

## 5. Training Data Flow

```
INPUT:
• config.yaml
• Dataset/HAM10000_metadata.csv
• Dataset/HAM10000_images_part_1/
• Dataset/HAM10000_images_part_2/

    ↓

[ 1. LOAD METADATA ]
  → pd.read_csv("HAM10000_metadata.csv")
  → DataFrame(10015, columns)

    ↓

[ 2. STRATIFIED SPLIT ]
  → Split at LESION level (not image)
  → Train: 70% (7000 lesions)
  → Val: 15% (1500 lesions)
  → Test: 15% (1500 lesions)
  → Maintain class distribution

    ↓

[ 3. DATALOADER (per batch) ]
  ┌─────────────────────────────┐
  │ For each image in batch:    │
  │ 1. Load image from disk     │
  │ 2. Resize to 224×224        │
  │ 3. Random augmentation:     │
  │    • Rotation ±15°          │
  │    • Flip H/V               │
  │    • Brightness ±10%        │
  │    • Contrast ±10%          │
  │ 4. Normalize (ImageNet μ/σ) │
  │ 5. Return (tensor, label)   │
  │ 6. Batch stack              │
  └─────────────────────────────┘
    → Tensor batch (32, 3, 224, 224)
    → Label batch (32,)

    ↓

[ 4. MODEL FORWARD PASS ]
  → output = model(batch_images)  ← (32, 7) logits

    ↓

[ 5. COMPUTE LOSS ]
  → loss_fn = CrossEntropyLoss(
      weight=[1.0, 6.0, 6.1, 13.0, 20.5, 47.3, 58.3]
    )
  → loss = loss_fn(output, labels)

    ↓

[ 6. BACKPROP & UPDATE ]
  → optimizer.zero_grad()
  → loss.backward()
  → optimizer.step()

    ↓

[ 7. VALIDATION CHECK (per epoch) ]
  → Evaluate on val_loader
  → Compute val_accuracy
  → If improved: save checkpoint

    ↓

[ 8. OUTPUT ]
  → Saved: models/best_model.pth
  → Logged: reports/training_log.csv


LOOP: Repeat for N epochs until:
  • Epochs reached, OR
  • Early stopping (val_loss ↑ 10 epochs)
```

---

## 6. Error Handling Paths

```
SCENARIO 1: Invalid File Format
┌─────────────────┐
│ User uploads    │
│ document.pdf    │
└────────┬────────┘
         │
    [ VALIDATE_UPLOAD ]
         │
    Extension check: .pdf != [jpg, png]
         │
         ├─ ERROR ──→ HTTP 400
         │
         └─ Message: "Please upload JPG or PNG image"
                     └─ Display to user


SCENARIO 2: File Too Large  
┌─────────────────┐
│ User uploads    │
│ image45MB.jpg   │
└────────┬────────┘
         │
    [ VALIDATE_UPLOAD ]
         │
    Size check: 45MB > 10MB limit
         │
         ├─ ERROR ──→ HTTP 413
         │
         └─ Message: "File too large (max 10MB)"


SCENARIO 3: Model Weights Missing
┌──────────────────────┐
│ User submits image   │
│ System tries to      │
│ load model           │
└────────┬─────────────┘
         │
    [ LOAD MODEL ]
         │
    File check: models/best_model.pth NOT FOUND
         │
         ├─ ERROR ──→ HTTP 503
         │
         └─ Message: "Model not ready, try again later"
            Actions: Log error, notify admin


SCENARIO 4: OOM During Inference
┌──────────────────────┐
│ 10000×10000 image    │
│ Resize fails         │
└────────┬─────────────┘
         │
    [ PREPROCESS_IMAGE ]
         │
    Memory error: Cannot allocate 800MB
         │
         ├─ CATCH EXCEPTION
         │
         ├─ Log error: "OOM on image: XYZ"
         │
         ├─ ERROR ──→ HTTP 413
         │
         └─ Message: "Image too large after resize"
```

---

## 7. Data Stores

| Data Store | Location | Purpose | Size | Access Pattern |
|-----------|----------|---------|------|---|
| **Metadata** | Dataset/HAM10000_metadata.csv | Image paths & labels | 3 MB | Read-only (Phase 3) |
| **Images** | Dataset/HAM10000_images_part_1/2/ | Raw training data | ~3.2 GB | Read-only (Phase 3) |
| **Model Weights** | models/ | Pre-trained CNN | ~200 MB | Read (Phase 6) |
| **Config** | config.yaml | All settings | 5 KB | Read-only (Phases 1-9) |
| **Logs** | logs/ | Session logs | ~50 MB/month | Write (all phases) |
| **Temp Files** | /tmp | Uploaded images | ~10 MB | Write/Delete (Phase 6) |
| **Results** | reports/ | Metrics & analysis | ~100 MB | Write (all phases) |

---

## 8. Control Flow Diagram

```
                        MAIN PROCESS
                            │
                ┌───────────┼───────────┐
                │           │           │
              CLI       WEB UI        TEST
            (train)    (predict)      (pytest)
              │           │             │
              ├─→ IMPORT MODULES
              │   • torch
              │   • flask
              │   • dataset
              │   • model
              │   • utils
              │
              ├─→ SETUP LOGGING
              │   Config: config.yaml
              │
              ├─→ LOAD CONFIG
              │   Read: config.yaml
              │
              ├─→ INITIALIZE COMPONENTS
              │   ├─ DatasetManager()
              │   ├─ CNNModel() / TransferLearningModel()
              │   ├─ FlaskApp()
              │   └─ Utilities
              │
              ├─ [If CLI]
              │  ├─ train.py
              │  │  ├─ Load dataset
              │  │  ├─ Build model
              │  │  ├─ Train loop
              │  │  └─ Save checkpoint
              │  │
              │  ├─ predict.py
              │  │  ├─ Load model
              │  │  ├─ Load image(s)
              │  │  ├─ Preprocess
              │  │  ├─ Infer
              │  │  └─ Print results
              │  │
              │  ├─ evaluate.py
              │  │  ├─ Load model
              │  │  ├─ Load test set
              │  │  ├─ Compute metrics
              │  │  └─ Generate report
              │  │
              │  └─ app.py
              │     └─ flask.run()
              │
              ├─ [If WEB]
              │  ├─ Listen on localhost:5000
              │  │
              │  ├─ GET / 
              │  │  └─ Serve upload form
              │  │
              │  ├─ POST /predict
              │  │  ├─ Validate
              │  │  ├─ Preprocess
              │  │  ├─ Infer
              │  │  └─ Return JSON
              │  │
              │  ├─ GET /results
              │  │  └─ Display prediction
              │  │
              │  ├─ Error handlers
              │  │  ├─ 400 Bad Request
              │  │  ├─ 413 Payload Too Large
              │  │  ├─ 500 Server Error
              │  │  └─ 503 Service Unavailable
              │  │
              │  └─ Shutdown gracefully
              │
              └─→ CLEANUP
                  ├─ Close connections
                  ├─ Flush logs
                  └─ Release memory
```

---

## 9. Latency Budget

| Component | Min | Typical | Max | Budget | Notes |
|-----------|-----|---------|-----|--------|-------|
| File upload | 10ms | 50ms | 200ms | 200ms | Network-dependent |
| Validation | 5ms | 10ms | 50ms | 50ms | Local filesystem |
| Image load | 20ms | 100ms | 500ms | 500ms | File I/O |
| Resize | 50ms | 200ms | 500ms | 500ms | PIL operations |
| Normalize | 20ms | 50ms | 100ms | 100ms | NumPy ops |
| **Preprocessing Total** | - | - | - | **1000ms** | - |
| Model load | 100ms | 500ms | 2000ms | 2000ms | First time only |
| Inference | 500ms | 1500ms | 3000ms | 3000ms | CPU-based |
| Softmax | 5ms | 20ms | 50ms | 50ms | GPU-fast |
| **Inference Total** | - | - | - | **4000ms** | - |
| JSON format | 10ms | 20ms | 50ms | 50ms | Python serialization |
| HTTP response | 10ms | 30ms | 100ms | 100ms | Network-dependent |
| **Response Total** | - | - | - | **150ms** | - |
| **TOTAL** | **155ms** | **1950ms** | **6950ms** | **≤5000ms** | ✅ SLA Met |

---

**DFD Status**: ✅ **COMPLETE & APPROVED**  
**Next Review**: Phase 3 (Implementation validation)

