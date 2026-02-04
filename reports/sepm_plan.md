## SEPM Project Development Plan

**Project Title**: Skin Cancer Disease Prediction System  
**Domain**: Medical Image Analysis / Deep Learning  
**Development Model**: Incremental / Iterative

---

## 1. Roles and Responsibilities

- **Project Manager (PM)**  
  - Plan schedule, track progress, manage risks, coordinate meetings, ensure documentation.
- **Lead Developer / ML Engineer**  
  - Design model architecture, implement training and evaluation, optimize performance.
- **Data Engineer**  
  - Manage datasets, preprocessing pipelines, augmentation, and data quality checks.
- **UI/UX Developer**  
  - Design and implement user interface (CLI / Flask web app).
- **Tester / QA Engineer**  
  - Prepare test plans, test cases (functional, non-functional), and perform validation.
- **Documentation Owner**  
  - Maintain SRS, design documents, SEPM artefacts, user manual, and final report.

> For a small student team, one person may hold multiple roles.

---

## 2. Work Breakdown Structure (WBS)

### Phase 1 – Inception & Requirements (Week 1–2)

- **Task 1.1**: Finalize problem statement and project objectives  
- **Task 1.2**: Collect and analyze reference papers (`References/`)  
- **Task 1.3**: Finalize SRS (functional and non-functional requirements)  
- **Task 1.4**: Define success criteria (target accuracy, latency, etc.)

### Phase 2 – Analysis & High-Level Design (Week 2–3)

- **Task 2.1**: Identify dataset (e.g., HAM10000) and its structure  
- **Task 2.2**: Design system architecture and module boundaries  
- **Task 2.3**: Define data flow (upload → preprocessing → model → results)  
- **Task 2.4**: Select technology stack (TensorFlow vs PyTorch, Flask vs CLI)

### Phase 3 – Dataset & Preprocessing Module (Week 3–4)

- **Task 3.1**: Implement `Dataset Manager` to load and validate data  
- **Task 3.2**: Implement preprocessing functions (resize `224×224`, normalize, denoise)  
- **Task 3.3**: Implement augmentation (rotation, flip, random crop/zoom)  
- **Task 3.4**: EDA notebooks (class balance, sample visualization)

### Phase 4 – Model Design & Baseline Implementation (Week 4–5)

- **Task 4.1**: Design baseline CNN architecture  
- **Task 4.2**: Implement training loop and loss/optimizer  
- **Task 4.3**: Implement evaluation metrics (accuracy, precision, recall, confusion matrix)  
- **Task 4.4**: Run baseline experiments and capture results in `notebooks/` and `reports/`

### Phase 5 – Model Improvement & Tuning (Week 5–6)

- **Task 5.1**: Introduce regularization (dropout, weight decay)  
- **Task 5.2**: Try transfer learning (e.g., ResNet, EfficientNet)  
- **Task 5.3**: Hyperparameter tuning (learning rate, batch size, epochs)  
- **Task 5.4**: Compare models and select best-performing one

### Phase 6 – Prediction API & UI (Week 6–7)

- **Task 6.1**: Implement prediction script (`predict.py`) for a single image  
- **Task 6.2**: Implement Flask-based web UI (upload form + result display)  
- **Task 6.3**: Integrate model loading with UI (ensure prediction < 5 seconds)  
- **Task 6.4**: Basic UI validation and usability refinement

### Phase 7 – Testing & Quality Assurance (Week 7–8)

- **Task 7.1**: Create test plan and test cases (FR1–FR13, NFRs)  
- **Task 7.2**: Unit tests for preprocessing, model I/O, and inference functions  
- **Task 7.3**: System and integration tests (end-to-end: upload → prediction)  
- **Task 7.4**: Performance tests (latency, memory, robustness to invalid images)

### Phase 8 – Deployment & Packaging (Week 8–9)

- **Task 8.1**: Freeze requirements in `requirements.txt`  
- **Task 8.2**: Provide run scripts / batch files for Windows  
- **Task 8.3**: Optional: Create Dockerfile or simple deployment guide  
- **Task 8.4**: Prepare demo dataset and sample images for evaluation

### Phase 9 – Documentation & Final Presentation (Week 9–10)

- **Task 9.1**: Finalize README, user manual, and technical documentation  
- **Task 9.2**: Finalize SEPM artefacts (SRS, design doc, test report, risk plan)  
- **Task 9.3**: Prepare slides and demo scenario  
- **Task 9.4**: Conduct internal rehearsal and refine demo

---

## 3. Milestones and Deliverables

- **M1 (End of Week 2)**: Approved SRS and initial architecture diagram  
- **M2 (End of Week 4)**: Dataset & preprocessing working; baseline CNN runs end-to-end  
- **M3 (End of Week 6)**: Tuned model achieving acceptable accuracy; documented results  
- **M4 (End of Week 7)**: UI integrated with prediction; end-to-end system functional  
- **M5 (End of Week 9)**: Deployment-ready package; completed documentation  
- **M6 (End of Week 10)**: Final presentation and demonstration delivered

---

## 4. Tools and Practices

- **Version Control**: Git + GitHub/GitLab  
- **Issue Tracking / PM**: GitHub Issues, Trello, or Jira  
- **Documentation**: Markdown files in `reports/` and `README.md`  
- **Coding Standards**: PEP8 for Python, type hints where possible  
- **Testing**: `pytest` or `unittest`  
- **Communication**: Weekly status meetings, progress reports in SEPM log

---

## 5. Risk Management

- **R1: Dataset imbalance / poor quality**  
  - *Impact*: Lower accuracy, biased predictions  
  - *Mitigation*: Use augmentation, class weighting, careful EDA  

- **R2: Insufficient accuracy by deadline**  
  - *Impact*: Fails success criteria  
  - *Mitigation*: Use transfer learning from pre-trained CNNs, simplify classification task (binary before multi-class)  

- **R3: Hardware limitations (no GPU)**  
  - *Impact*: Long training times  
  - *Mitigation*: Use smaller models, lower resolution during training, or train on external GPU/cloud and deploy locally  

- **R4: Scope creep** (e.g., adding mobile app too early)  
  - *Impact*: Deadlines missed  
  - *Mitigation*: Keep mobile/cloud features as **future enhancement** after core milestones

---

## 6. Quality Assurance Strategy

- **Code Quality**: Code reviews within team, linting (`flake8`/`black`)  
- **Validation Strategy**: Train/validation/test split; report confusion matrices and per-class metrics  
- **Traceability**: Map each functional requirement (FR1–FR13) to implementation modules and test cases in a **traceability matrix**

