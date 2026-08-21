# Portfolio Project Checklist — Music Taste Recommender

Every project must check every box before it counts as "portfolio-ready" and before
work moves on to the next project. Derived directly from the ML-portfolio hiring
guidelines pasted 2026-08-21 (full source discussion in
`Desktop/PORTFOLIO-PROJECT-PLANS/portfolio_rubric_and_roadmap.md`).

Updated 2026-08-21 after the deployment/Docker/MLflow retrofit — every checked box below
was actually run and verified during that retrofit, not marked aspirationally.

## 1. End-to-End Implementation
- [x] Data preprocessing pipeline present and documented — `notebooks/01_data_audit.ipynb`, `src/data_loading.py`
- [x] Feature engineering step present and documented — `src/content_based.py` (StandardScaler over audio features)
- [x] Training pipeline present — `notebooks/03_collaborative.ipynb`, `scripts/train_als_with_mlflow.py`
- [x] Evaluation step present, with real metrics — `src/evaluation.py`, precision@10/recall@10 verified
- [x] Deployed: runs somewhere beyond a notebook — FastAPI (`api/`) and Streamlit (`app/`), both verified running locally and in Docker
- [x] Monitoring present, or at minimum a stated monitoring plan — README "Monitoring" section

## 2. Reproducibility
- [x] Clean, structured codebase (not notebooks-only)
- [x] `requirements.txt` included
- [x] A stranger can clone and run it without DMing the author — verified: moved the local model aside, confirmed `src/artifacts.py` downloads it fresh from Hugging Face Hub and loads correctly

## 3. Business Relevance
- [x] Tied to a practical/business outcome, not just a metric — README "Business relevance" section
- [x] That outcome framing is stated explicitly in the README

## 4. Scalability Awareness
- [x] Latency, cost, and infrastructure constraints are discussed — README "Scalability considerations"
- [x] A stated plan for how it would scale beyond the demo

## 5. Documentation & Storytelling
- [x] README has a problem statement
- [x] README has an architecture diagram — Mermaid diagram, renders natively on GitHub
- [x] README has results
- [x] README has "how to run" instructions — covers API, Streamlit, Docker, and MLflow tracking
- [ ] (Bonus) a blog post or demo video — not done, optional

## 6. Modern Stack
- [x] Docker — built and run-tested (hit and fixed a real `libgomp1` missing-dependency bug in the process)
- [x] Experiment tracking (MLflow) — real run logged: params, precision@10/recall@10 metrics, model artifact
- [x] Hugging Face — model artifacts hosted on Hugging Face Hub (`ChrisOmara/music-taste-recommender-artifacts`)

## 7. Recruiter-Scan Signals
- [x] README is understandable in 30 seconds
- [ ] Commit history reads as polished work — pending: this retrofit's changes need a clean commit
- [ ] Live deployment link (Streamlit Community Cloud) — pending: needs the user's own one-click "connect repo" step on streamlit.io, tied to their account
- [x] Architecture diagram a non-technical reviewer can follow

## 8. Elevation Pass
- [x] Data augmentation/preprocessing pipeline
- [x] MLOps practices: model versioning, experiment tracking
- [x] Deployed as a REST API (FastAPI) — `/health`, `/users/sample`, `/tracks/sample`, `/recommend/{user_id}`, `/similar/{track_id}`, all verified with real responses
- [x] Demo app (Streamlit) — verified boots and serves both flows (recommend-for-listener, find-similar-track)
- [x] Written explanation of a real-world application — README "Business relevance"

## 9. Forward-Looking Signals (2025+)
- [x] Any LLM use is responsible and cost-aware — n/a, no LLM component in this project
- [x] Bias/explainability/compliance trade-offs considered — case study is explicit about the cold-start limitation, join-coverage limitation, and what each honestly means
- [x] Multimodal awareness noted where relevant — n/a, single-modality (audio features + implicit feedback), not claimed otherwise
- [x] Project is treated as living — this retrofit itself is the evidence
