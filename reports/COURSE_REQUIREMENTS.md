# Coverage of the supplied university instructions

Source reviewed: `/home/coonfituuree/Downloads/Machine_Learning_Final_Project_Instructions.pdf`, “MLA Final Project Instructions”. This maps the PDF's six workflow sections and grading rubric to concrete deliverables. The software and written materials support all technical requirements; students remain responsible for their submission, individual contributions, and live defense.

| PDF requirement / rubric weight | Concrete implementation and evidence |
|---|---|
| Problem formulation and motivation (10%) | README and methodology define recruitment-fraud screening, binary labels, intended English text inputs, output/score, human-review purpose, and success metrics. |
| Data origin, dimensions, usage restrictions | Official Kaggle EMSCAD archive, byte equality with supplied CSV, SHA-256, public metadata/license, paper citation in `data/` and methodology. |
| EDA: distributions, missing values, outliers, class balance, relationships (15%) | `training/prepare.py`, `reports/eda.json`, `reports/figures/eda.png`; methodology explains imbalance, length tails, profile missingness, geography and modeling consequences. |
| Data limitations, sampling bias, privacy | Methodology discusses historical/platform/geography scope, text-only authentication limits, local ephemeral uploads, dataset excerpts and label limitations. |
| Preparation and evaluation design (15%) | Shared HTML/whitespace/missing-value handling, label/ID exclusion, exact deduplication, conflict quarantine, near-duplicate audit, profile/description connected groups, saved deterministic group splits, training-only vectorizer. |
| Address class imbalance | Training-only balanced LR weights and neural positive weight; no SMOTE or synthetic ads. |
| Baseline | Actually fitted majority-class DummyClassifier, same test partition and saved artifact. |
| Non-neural ML model | Actually trained TF-IDF + Logistic Regression; three-value C search; selected estimator and metrics saved. |
| Neural network implemented and trained (30%) | Original PyTorch `TfidfMLP`, actual minibatch training with backpropagation; state_dict, search logs, loss/AP histories, selected checkpoint saved. This is not pretrained-only inference. |
| Architecture, activations, loss, optimizer, batch size, epochs, regularization | Versioned configuration and methodology detail 12,000→128→1, ReLU, dropout .5, weighted BCEWithLogitsLoss, AdamW lr .001/weight decay .0001, batch 128, epoch limit 30, AP early stopping and epoch 21 checkpoint. |
| Hyperparameter tuning and consistent comparison | `reports/search.json`, validation-only AP selection and F1 thresholds; all models use shared TF-IDF and identical grouped holdout. |
| Evaluation, interpretation and critical analysis (20%) | Comparison JSON/CSV, fraud precision/recall/F1/AP/ROC-AUC, confusion/PR figures, training curves, real errors and successes, LR-only coefficient explanation, evidence-based overfitting discussion and next steps. |
| Reproducible code and environment | Isolated Python 3.12, exact lockfiles, dataset script/manual placement, seed 42, organized modules, setup/start commands, artifact integrity checks, test suite. |
| Prediction for a new example | Real saved-model FastAPI inference, interactive React UI and file upload, browser flow verification through Vite proxy. |
| Presentation and defense (10%) | `presentation.pptx`, `DEFENSE_NOTES.md`, methodology and working local demo. Students must review slides, add names/actual contributions, follow the PDF's no-phone defense rule, and explain the implementation independently. |
| Cite data, code, pretrained models, AI assistance | Dataset/paper and library citations plus explicit Codex assistance disclosure in README, methodology and slides. No pretrained weights. |
| Group requirements | No invented team names or contributions. The PDF allows at most three students; actual member contributions must be supplied by students. |

Local deployment is optional in the PDF and is implemented as a native development application. Public hosting, HTTPS and domains were explicitly excluded by the user. No external classification API is used.
