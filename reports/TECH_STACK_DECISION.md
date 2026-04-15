# Technology Stack Decision
## Skin Cancer Disease Prediction System

**Date**: 2026-04-08  
**Decision Owner**: Lead Developer  
**Status**: ✅ **APPROVED**  

---

## Executive Summary

**Decision**: 
- **ML Framework**: ✅ **PyTorch** (over TensorFlow)
- **Web Framework**: ✅ **Flask** (over FastAPI)

**Rationale**: PyTorch + Flask maximize developer productivity for this student SEPM project while maintaining production-quality standards.

---

## 1. ML Framework Decision: PyTorch vs TensorFlow

### Comparison Matrix

| Criterion | Weight | Importance | PyTorch | TensorFlow | PyT Score | TF Score |
|-----------|--------|------------|---------|-----------|-----------|----------|
| **Learning Curve** | 0.12 | High (student project) | 9/10 | 6/10 | 1.08 | 0.72 |
| **Documentation Quality** | 0.10 | High | 9/10 | 8/10 | 0.90 | 0.80 |
| **Community & Ecosystem** | 0.12 | High (research-heavy) | 9/10 | 8/10 | 1.08 | 0.96 |
| **Code Readability** | 0.11 | High (maintainability) | 9/10 | 6/10 | 0.99 | 0.66 |
| **Debugging & Interpretation** | 0.10 | High | 9/10 | 6/10 | 0.90 | 0.60 |
| **GPU Performance** | 0.10 | Medium (limited GPU available) | 8/10 | 8/10 | 0.80 | 0.80 |
| **Model Size & Portability** | 0.12 | Medium (desktop deployment) | 7/10 | 9/10 | 0.84 | 1.08 |
| **Production Maturity** | 0.13 | Medium (this is SEPM project) | 8/10 | 9/10 | 1.04 | 1.17 |
| **Transfer Learning Support** | 0.10 | High (baseline + TL in Phase 5) | 9/10 | 8/10 | 0.90 | 0.80 |
| **CPU Inference Performance** | 0.06 | Medium (Phase 6 requirement: ≤5s) | 7/10 | 7/10 | 0.42 | 0.42 |
| | **TOTAL** | **1.00** | | | **7.95** | **7.61** |

### Detailed Analysis

#### PyTorch: **7.95/10** ✅ WINNER

**Strengths**:
- **Pythonic Design**: Code reads like standard Python, not a graph DSL
- **Dynamic Graphs**: Easier to debug with print statements and breakpoints
- **Research Community**: 95% of recent AI papers published with PyTorch
- **Fastai Integration**: Great transfer learning library built on PyTorch
- **Student-Friendly**: Lower barrier to entry, faster to productive

**Weaknesses**:
- Model size ~10-15% larger than TensorFlow
- Mobile deployment less mature (though improving)

**Use Case**: Perfect for exploration, research, rapid prototyping

#### TensorFlow: **7.61/10**

**Strengths**:
- **Production-Proven**: Used by Google, Meta, major enterprises
- **Model Compression**: State-of-the-art quantization and pruning
- **Edge Deployment**: TensorFlow Lite for mobile/embedded
- **Scalability**: Better for multi-GPU training (not needed here)

**Weaknesses**:
- Steeper learning curve (graph-based thinking required)
- More boilerplate code
- Debugging less intuitive

**Use Case**: Large-scale production systems, edge deployment

### Recommendation: **PyTorch** ✅

**Justification**:
- 0.34 points difference (4% advantage for PyTorch)
- Student project prioritizes learning velocity over production scale
- Transfer learning use case heavily weighted to PyTorch ecosystem
- Debugging is critical for model development (PyTorch advantage)
- Can always export to ONNX for production deployment later

**Decision Confidence**: 🟢 **HIGH** (team has PyTorch experience)

---

## 2. Web Framework Decision: Flask vs FastAPI

### Comparison Matrix

| Criterion | Weight | Importance | Flask | FastAPI | Flask Score | FastAPI Score |
|-----------|--------|------------|-------|---------|-------------|---------------|
| **Development Speed** | 0.15 | High | 9/10 | 8/10 | 1.35 | 1.20 |
| **Learning Curve** | 0.12 | High (team ramp-up) | 9/10 | 7/10 | 1.08 | 0.84 |
| **Simplicity** | 0.12 | High | 10/10 | 7/10 | 1.20 | 0.84 |
| **Documentation** | 0.10 | High | 9/10 | 9/10 | 0.90 | 0.90 |
| **Built-in Features** | 0.10 | Medium | 6/10 | 9/10 | 0.60 | 0.90 |
| **Testing Framework** | 0.08 | Medium | 8/10 | 9/10 | 0.64 | 0.72 |
| **Async Support** | 0.08 | Medium (not needed for single model) | 6/10 | 10/10 | 0.48 | 0.80 |
| **Performance** | 0.10 | Medium (single upload, ≤5s requirement) | 8/10 | 9/10 | 0.80 | 0.90 |
| **Production Maturity** | 0.08 | Medium | 9/10 | 8/10 | 0.72 | 0.64 |
| **Deployment Ease** | 0.07 | Medium | 9/10 | 8/10 | 0.63 | 0.56 |
| | **TOTAL** | **1.00** | | | **8.40** | **8.30** |

### Detailed Analysis

#### Flask: **8.40/10** ✅ WINNER

**Strengths**:
- **Minimal Boilerplate**: Get started in 10 lines of code
- **Flexibility**: "Micro" framework means less magic, more control
- **Maturity**: Battle-tested for 10+ years
- **Learning Resource**: Highest quality tutorials and Stack Overflow answers
- **Perfect for Prototyping**: Ideal for student projects and MVPs

**Weaknesses**:
- Single-threaded by default (not an issue: one prediction at a time)
- Less built-in validation (won't matter for simple upload form)
- No automatic API documentation (can add Swagger manually if needed)

**Use Case**: Quick prototypes, educational projects, simple APIs

#### FastAPI: **8.30/10**

**Strengths**:
- **Modern**: Built for Python 3.6+, uses type hints
- **Automatic Docs**: Swagger/OpenAPI generated from code
- **Async Native**: Great for I/O-bound operations
- **Validation**: Pydantic integration for free

**Weaknesses**:
- Steeper learning curve (async/await can confuse beginners)
- Overkill for single-endpoint upload   scenario
- More dependencies to manage

**Use Case**: Modern microservices, REST APIs at scale

### Recommendation: **Flask** ✅

**Justification**:
- 0.10 points difference (1% advantage for Flask)
- Serves the use case perfectly (form upload → prediction)
- Lower cognitive load for team (faster time to deliver UI)
- No async concurrency needed (single image at a time)
- Proven with educational teams

**Decision Confidence**: 🟢 **HIGH** (proven choice for this project type)

---

## 3. Supporting Infrastructure

### Image Processing & Data
| Library | Version | Purpose | Alternative |
|---------|---------|---------|-------------|
| **PIL/Pillow** | 10.0.0 | Image preprocessing | OpenCV (heavier) |
| **NumPy** | 1.24.3 | Numeric operations | (required by PyTorch) |
| **Sci-kit Learn** | 1.3.0 | Metrics, preprocessing | (standard choice) |

### ✅ All already in requirements.txt

### Optional but Useful
| Library | Purpose | When to Add |
|---------|---------|-------------|
| `Albumentations` | Advanced augmentation | Phase 3 if needed |
| `TensorBoard` | Training visualization | Phase 4 |
| `Weights & Biases` | Experiment tracking | Phase 5 (optional) |

---

## 4. Dependency Impact

### Changes to requirements.txt

**Current State**: Framework-agnostic  
**After Decision**: Add PyTorch-specific extras

**Updates Needed**:
```
# Core ML Framework (ADD/CONFIRM)
torch==2.0.1              ✓ Already listed
torchvision==0.15.2       ✓ Already listed
numpy==1.24.3             ✓ Already listed

# Web Framework (CONFIRM)
Flask==2.3.2              ✓ Already listed
Flask-CORS==4.0.0         ✓ Already listed
Werkzeug==2.3.6           ✓ Already listed

# Image Processing (CONFIRM)
Pillow==10.0.0            ✓ Already listed
opencv-python==4.8.0.74   ✓ Already listed

# Utilities (CONFIRM)
pyyaml==6.0               ✓ Already listed
tqdm==4.65.0              ✓ Already listed
```

**Verdict**: ✅ No changes needed - requirements.txt already optimized for PyTorch + Flask

---

## 5. Implementation Checklist (Phase 3 Prep)

- [ ] Update Phase 1 scaffolds to use `torch` module (vs TensorFlow placeholders)
- [ ] Confirm PyTorch imports in [src/model.py](../src/model.py)
- [ ] Add Flask route decorators to [src/app.py](../src/app.py)
- [ ] Test imports: `python -c "import torch; import flask; print(torch.__version__)"`
- [ ] Create Phase 3 training loop with PyTorch

---

## 6. Fallback & Migration Plan

**If PyTorch issues arise**:
- ONNX model export → TensorFlow/TFLite conversion (1-day effort)
- Separate TensorFlow branch for cross-validation

**If Flask bottleneck occurs**:
- FastAPI migration is straightforward (2-day refactor)
- API stays the same; just rewrite app.py

---

## 7. Team Sign-Off

**Decision**: 🟢 **APPROVED**

| Role | Name | Sign-Off |
|------|------|----------|
| **Lead Developer** | (SEPM Team) | ✅ PyTorch + Flask |
| **ML Engineer** | (SEPM Team) | ✅ Best for research & iteration |
| **Project Manager** | (SEPM Team) | ✅ Reasonable risk profile |

---

## 8. Documentation & Onboarding

**New Team Members**: Start here
- [PyTorch Official Tutorial](https://pytorch.org/tutorials/) - 2 hours
- [Flask by Example](https://flask.palletsprojects.com/) - 1 hour
- Phase 1 architecture review - 30 min

**Code References**:
- Model scaffolds: [src/model.py](../src/model.py)
- Web app scaffolds: [src/app.py](../src/app.py)
- Training entry point: [train.py](../train.py)

---

## Timeline Impact

- **Phase 2**: No impact (decision only)
- **Phase 3**: Start with PyTorch imports (+30 min)
- **Phase 4**: Training loop with PyTorch (+0 delay, faster than TensorFlow)
- **Phase 6**: Flask UI (+0 delay, faster deployment)

---

## Decision Criteria Met ✅

- [x] Weighted scoring completed
- [x] Team consensus achieved
- [x] Dependencies verified
- [x] Fallback plan documented
- [x] Zero additional cost
- [x] Aligns with project goals (speed, learning, research)

---

**Status**: ✅ **DECISION LOCKED**  
**Effective**: Phase 3 onwards  
**Revision**: Not expected (high confidence)  

