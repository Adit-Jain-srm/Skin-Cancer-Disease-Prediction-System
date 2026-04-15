# Phase 2: Analysis & High-Level Design - Execution Plan
## Skin Cancer Disease Prediction System

**Phase**: 2/9  
**Duration**: Week 2–3 (2026-04-09 to 2026-04-21)  
**Target Milestone**: M2 (End Week 4): Dataset loaded + baseline CNN functional  
**Status**: 📋 **PLANNING**

---

## Execution Strategy

### Approach
1. **Sequential Tasks with Verification**: Each task builds on previous
2. **Subagent Usage**: Offload dataset exploration to subagent to preserve context
3. **Modular Verification**: Test each output before proceeding
4. **Documentation-Driven**: All decisions documented with rationale

### Risk Mitigation
- **Dataset issues**: Subagent validates data quality early
- **Architecture confusion**: Create visual diagram before coding
- **Tech stack indecision**: Use comparison table with weighted criteria

---

## Task Breakdown & Specs

### Task 2.1: Analyze HAM10000 Dataset Structure 🔍

**Objective**: Understand dataset format, quality, class distribution  
**Owner**: Data Engineering team  
**Deliverables**: 
- Dataset analysis report in `reports/PHASE2_DATASET_ANALYSIS.md`
- Sample visualization in `notebooks/01_eda.ipynb`

**Steps**:
1. Load `Dataset/HAM10000_metadata.csv` 
2. Check structure: columns, rows, data types
3. Analyze class distribution (7 classes)
4. Verify image counts match metadata
5. Check for missing/corrupted images
6. Identify preprocessing requirements (resolutions, formats)
7. Document dataset constraints in report

**Success Criteria**:
- ✅ Metadata loads without errors
- ✅ Class distribution visualized
- ✅ All images accessible (no 404s)
- ✅ Resolution statistics computed
- ✅ Report includes data quality score

**Estimated Duration**: 2 hours  
**Blocker Dependencies**: None (Dataset already in repo)

---

### Task 2.2: Design System Architecture Diagram 📐

**Objective**: Create clear visual of module interactions  
**Owner**: Lead Developer  
**Deliverables**:
- Architecture diagram in `reports/ARCHITECTURE.md` (ASCII + Mermaid)
- Module interface specs in `reports/ARCHITECTURE_SPEC.md`

**Steps**:
1. Review Phase 1 module design (src/)
2. Create 3 diagrams:
   - **Layer diagram**: Data → Model → App → Utils
   - **Sequence diagram**: User upload → predict → display flow
   - **Module interface diagram**: Show inputs/outputs per module
3. Document responsibility matrix (who owns what)
4. Define clear interfaces between layers

**Success Criteria**:
- ✅ All 5 modules represented
- ✅ Data flow clear and testable
- ✅ Interface contracts defined
- ✅ Diagram matches code structure

**Estimated Duration**: 3 hours  
**Blocker Dependencies**: Task 2.1 (optional, for context)

---

### Task 2.3: Update Data Flow Diagram (DFD) 🔄

**Objective**: Create detailed data flow from user to result  
**Owner**: System Analyst  
**Deliverables**:
- Enhanced `reports/DFD.md` with swimlanes
- Updated `reports/DFD.png` (visual)

**Steps**:
1. Define swimlanes: User, UI, Preprocessing, Model, Results
2. Trace complete flow:
   - User uploads image
   - Flask validates format
   - DatasetManager preprocesses
   - CNN model infers
   - Results returned to UI
3. Add timing annotations (~5s SLA)
4. Identify error paths (invalid image handling)
5. Document data stores (model weights, temp files)

**Success Criteria**:
- ✅ All swimlanes present
- ✅ All data transformations shown
- ✅ Error handling paths visible
- ✅ Latency targets annotated

**Estimated Duration**: 2 hours  
**Blocker Dependencies**: Task 2.2 (architecture must be clear first)

---

### Task 2.4: Select Technology Stack 🛠️

**Objective**: Make PyTorch vs TensorFlow and Flask vs FastAPI decisions  
**Owner**: Tech Lead  
**Deliverables**:
- Decision document in `reports/TECH_STACK_DECISION.md`
- Rationale + comparison table
- Dependency updates in `requirements.txt` (if needed)

**Criteria Matrix**:
| Criterion | Weight | PyTorch | TensorFlow | Score PyT | Score TF |
|-----------|--------|---------|-----------|-----------|----------|
| Learning curve | 0.15 | Moderate | Steep | 8 | 6 |
| Prod maturity | 0.20 | High | Very High | 8 | 9 |
| Mobile support | 0.10 | Good | Excellent | 7 | 9 |
| Doc quality | 0.15 | Excellent | Good | 9 | 7 |
| Community | 0.15 | Large | Very Large | 8 | 9 |
| Flexibility | 0.15 | High | Medium | 9 | 7 |
| GPU efficiency | 0.10 | Very Good | Very Good | 8 | 8 |
| **TOTAL** | 1.0 | - | - | **8.1** | **7.8** |

**Decision Logic**:
- If weighted score ≥ 8.0: Recommend that framework
- If within 0.3 points: Choose by team preference + available expertise

**Success Criteria**:
- ✅ Weighted comparison completed
- ✅ Final decision documented
- ✅ Reasoning clear for all stakeholders
- ✅ Dependencies updated if changed

**Estimated Duration**: 1 hour  
**Blocker Dependencies**: None

---

## Verification Plan

### Pre-Completion Checks

**After Task 2.1** (Dataset Analysis):
```
[ ] Metadata CSV loads in Python
[ ] No column mismatches
[ ] Images on disk match metadata count
[ ] Class distribution shows no extreme imbalance
[ ] Report contains actionable insights
```

**After Task 2.2** (Architecture):
```
[ ] Diagram shows all 5 modules
[ ] Interfaces are defined (input/output types)
[ ] Data flow makes sense (no circular deps)
[ ] Each module has clear responsibility
[ ] Code structure matches diagram
```

**After Task 2.3** (DFD Update):
```
[ ] Swimlanes cover all actors/systems
[ ] Data stores identified
[ ] Latency annotations present
[ ] Error paths documented
[ ] Matches Phase 1 use cases
```

**After Task 2.4** (Tech Stack):
```
[ ] Comparison table weights sum to 1.0
[ ] Scores justified with evidence
[ ] Decision clear and defensible
[ ] Team agrees on choice
[ ] Dependencies updated
```

---

## Execution Order & Dependencies

```
Task 2.1 (Dataset)          [START]
    ↓ (output feeds into Design)
Task 2.2 (Architecture)     [PARALLEL OK]
    ↓ (must be complete before 2.3)
Task 2.3 (DFD)              
    
Task 2.4 (Tech Stack)       [INDEPENDENT - can run in parallel]

All complete → M2 Report → Phase 3 Ready
```

**Recommended Execution**:
1. Start with **Task 2.4** (Tech Stack) - independent, 1 hour
2. Run **Task 2.1** (Dataset) in subagent while you do Task 2.2
3. Complete **Task 2.2** (Architecture)
4. Run **Task 2.3** (DFD) with Task 2.2 output
5. Verify all outputs
6. Generate M2 Report

---

## Acceptance Criteria (Phase 2 Complete)

**Must Have**:
- ✅ Dataset analysis report with data quality assessment
- ✅ Architecture diagram showing all 5 modules
- ✅ DFD with swimlanes and error paths
- ✅ Technology stack decision documented
- ✅ Phase 2 completion report signed off

**Should Have**:
- ✅ EDA notebook with visualizations
- ✅ Module interface specification
- ✅ Risk assessment for chosen stack

**Nice to Have**:
- ✅ Performance benchmarks for framework choices
- ✅ Migration path documentation

---

## Timeline

| Task | Estimated | Scheduled | Owner |
|------|-----------|-----------|-------|
| 2.1 Dataset | 2h | 2026-04-09 to 04-10 | Data Team |
| 2.2 Architecture | 3h | 2026-04-10 to 04-11 | Lead Dev |
| 2.3 DFD | 2h | 2026-04-11 to 04-12 | Analyst |
| 2.4 Tech Stack | 1h | 2026-04-09 | Tech Lead |
| Verification | 1h | 2026-04-12 | QA |
| **M2 Report** | 2h | 2026-04-13 to 04-14 | PM |
| **Total** | **11h** | **Week 2** | - |

---

## Escalation & Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Dataset corrupted | Low | High | Subagent validates early; fallback to HMNIST |
| Architecture confusion | Low | Medium | Create multiple diagram formats |
| Tech stack deadlock | Low | Medium | Use weighted scoring matrix |
| Timeline slip | Medium | Medium | Run tasks in parallel; timebox decisions |

---

## Handoff to Phase 3

**At Phase 2 Completion**, Phase 3 receives:
- ✅ Clean dataset (validated, path verified)
- ✅ Architecture design (all modules clear)
- ✅ Tech stack chosen (ready to code)
- ✅ DFD approved (test cases derivable)
- ✅ Success criteria for P3 defined

**Phase 3 Entry Checklist**:
- [ ] Dataset loads without errors
- [ ] Architecture diagram approved
- [ ] Tech stack decision documented
- [ ] Team trained on selected frameworks
- [ ] Scaffolds modified for chosen stack

---

## Status: 📋 READY

**Next Action**: Begin Task 2.1 (Dataset Analysis) using subagent  
**Target Completion**: 2026-04-14  
**Gatekeeper**: M2 Milestone sign-off before Phase 3 starts
