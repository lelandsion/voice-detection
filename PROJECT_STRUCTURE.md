# Voice Verification With Noise (Live Demo) — Project Structure

> Note: Keep all tasks unassigned by default. Only the Project itself should be assigned to the requester and Leland.

## GitHub Project (1 board)
- **Name:** Voice Verification With Noise (Live Demo)
- **Owners:** Requester + Leland
- **Scope:** Single-user enrollment and yes/no verification under music/environmental noise; GPU allowed for training/embedding extraction; must ship live mic demo (enroll → verify with random prompts, VAD, embedding, score, decision).
- **Definition of Done (overall):**
  - Reproducible pipeline (scripts + configs)
  - ISMIR-style report (15–20 references)
  - Live Windows laptop mic demo with clear accept/reject behavior

## Milestones (deliverables)
- **M1 — Design/Requirements Spec** — Target: **Feb 7, 2026**  
  Deliverable: PDF + sources (Overleaf or paper/).  
  Acceptance: requirements (threat model, constraints, metrics EER/minDCF, FAR/FRR), demo requirements, tools/datasets (provisional ok), timeline + roles, related work 15–20 refs.  
  Owner: **Member 3 (integration); all contribute refs**

- **M2 — Data + Mixing Pipeline** — Target: **Feb 21, 2026**  
  Deliverable: dataset manifests + mixer script + reproducible splits.  
  Acceptance: mixes clean speech with music/noise at SNR bins (10/5/0/-5 dB), trial lists genuine vs impostor, deterministic with seed, logs metadata.  
  Owner: **Member 1**

- **M3 — Baseline v1: MFCC + Cosine + Evaluation Harness** — Target: **Mar 7, 2026**  
  Deliverable: baseline scores + ROC/DET/EER plots (≥2 SNR conditions).  
  Acceptance: enrollment builds template; verification outputs score + threshold decision; evaluation script saves EER + ROC/DET figures.  
  Owner: **Member 2 (baseline) + Member 3 (metrics/plots)**

- **M4 — Baseline v2: ECAPA/x-vector (GPU) + Scoring** — Target: **Mar 21, 2026**  
  Deliverable: embedding-based pipeline + comparison vs MFCC.  
  Acceptance: embeddings cached to disk; cosine (optional PLDA); shows improvement or explains why not.  
  Owner: **Member 2**

- **M5 — Live Microphone Demo (Required)** — Target: **Apr 4, 2026**  
  Deliverable: enroll/verify CLI that records from mic and prompts random words.  
  Acceptance: Windows mic capture working; VAD trims silence; handles too-short/noisy/low-confidence; latency measured end-to-end.  
  Owner: **Member 3 (demo) + Member 2 (model integration)**

- **M6 — Final Report + Reproducibility Package** — Target: **Apr 18, 2026**  
  Deliverable: final ISMIR paper + reproducible commands.  
  Acceptance: results tables/plots by SNR + noise type; clear protocol + limitations; one-command evaluation from prepared data.  
  Owner: **Member 3 (paper integration) + all review**

## Issues to create (grouped by milestone)
- **M1 — Design/Requirements Spec**
  - (paper) Set up ISMIR paper skeleton + section outline — Assignee: Member 3
  - (paper) Write Requirements: task definition + threat model + metrics + demo constraints — Assignee: Member 3
  - (paper) Related Work: collect 20 candidate refs, finalize 15–20 BibTeX — Assignees: All (split 7/7/6)
  - (paper) Tools/Datasets table + licensing notes — Assignee: Member 1
  - (paper) Timeline + roles table — Assignee: Member 3

- **M2 — Data + Mixing**
  - (data) Choose speech dataset(s) + impostor protocol — Assignee: Member 1
  - (data) Implement mixer: speech + music/noise, target SNR, random offsets — Assignee: Member 1
  - (data) Create manifests + train/dev/test splits + trial list generator — Assignee: Member 1
  - (infra) Data download/prep docs (Windows commands, folder structure) — Assignee: Member 1

- **M3 — MFCC baseline + evaluation**
  - (model) MFCC feature extraction module — Assignee: Member 2
  - (model) MFCC enrollment + cosine scoring baseline — Assignee: Member 2
  - (eval) Metrics: ROC/DET + EER + minDCF — Assignee: Member 3
  - (eval) Stratified evaluation by SNR/noise type + plotting — Assignee: Member 3
  - (paper) Add Baseline Experiment description to paper — Assignee: Member 3

- **M4 — ECAPA/x-vector baseline (GPU)**
  - (model) Select embedding stack (SpeechBrain ECAPA vs x-vector) + dependency lock — Assignee: Member 2
  - (model) Embedding extraction + caching + batch inference — Assignee: Member 2
  - (model) Scoring: cosine + optional PLDA — Assignee: Member 2
  - (eval) Compare MFCC vs embeddings; generate paper-ready tables — Assignees: Member 2 + Member 3

- **M5 — Live microphone demo (required)**
  - (demo) Mic recording module (Windows) + save WAV — Assignee: Member 3
  - (demo) Prompt generator (random words) + UX flow — Assignee: Member 3
  - (demo) VAD integration + “too short” handling — Assignee: Member 3
  - (demo/model) End-to-end CLI: enroll and verify using best model — Assignees: Member 3 + Member 2
  - (demo/eval) Latency measurement + demo checklist script — Assignee: Member 3

- **M6 — Final paper + reproducibility**
  - (infra) Config system (YAML/JSON) + run logging — Assignee: Member 2
  - (infra/eval) One-command evaluation pipeline (from manifests → metrics/figures) — Assignees: Member 1 + Member 3
  - (paper) Final write-up: methods/experiments/results/limitations — Assignee: Member 3 (all review)

## Labels to create
- `paper`
- `data`
- `model`
- `eval`
- `demo`
- `infra`
- `good-first-issue`

## Roles (reference)
- **Member 1 — Data/Protocol Lead:** datasets, mixing, manifests, trial generation, data docs
- **Member 2 — Modeling Lead (GPU):** MFCC baseline + ECAPA/x-vector pipeline, scoring, configs
- **Member 3 — Demo + Paper/Eval Lead:** mic demo, VAD, metrics/plots, paper integration
