# Voice Verification With Noise (Live Demo) — Project Structure

> Note: Keep all tasks unassigned by default in GitHub. The names below are **suggested owners** (role holders). Only the Project itself should be assigned to the requester and Leland.

## GitHub Project (1 board)
- **Name:** Voice Verification With Noise (Live Demo)
- **Owners:** Requester + Leland
- **Scope:** Single-user enrollment and yes/no verification under music/environmental noise; GPU allowed for training/embedding extraction; must ship live mic demo (enroll → verify with random prompts, VAD, embedding, score, decision).
- **Definition of Done (overall):**
  - Reproducible pipeline (scripts + configs)
  - ISMIR-style report (15–20 references)
  - Live Windows laptop mic demo with clear accept/reject behavior
 
## M1: Project Design Specifications (Completed)
- Describe tools and datasets – Liam
- 6 references (tools, datasets, literature) – Liam
- Related work section – Liam
- Timeline (Gantt chart) – Leland
- 6 references (tools, datasets, literature) – Leland
- Create GitHub project structure (milestones & issues) – Leland
- Describe proposed project – Lily
- 6 references (tools, datasets, literature) – Lily

## M2: Data Extraction
- Split data into training and test sets – Leland
- Preprocess data to begin with a start frequency – Leland
- Develop underlying logic to record sound (mic input) – Liam
- Process input sound using DFT or FFT – Lily
- Normalize output from processed input – Lily
- Develop logic to pass normalized input to model for binary classification – Lily
- Create end-to-end data extraction pipeline script – Leland

## M3: Insight Extraction
- Define evaluation metrics (FAR, FRR, EER, ROC) – Leland
- Perform exploratory data analysis on audio/features – Liam
- Analyze feature/embedding separability (same vs different speaker) – Liam
- Evaluate performance across noise conditions – Leland

## M4: Model Training
- Select baseline verification model architecture – Liam
- Implement training loop for binary speaker verification – Lily
- Train baseline model on clean data – Leland
- Save trained model checkpoints and configs – Leland

## M5: Noise and Robustness
- Add noise augmentation (music & environmental) – Liam
- Retrain or fine-tune model with noisy data – Lily
- Evaluate robustness under different SNR levels – Leland
- Compare clean vs noisy performance – Liam

## M6: Live Demo Integration
- Implement voice enrollment pipeline – Liam
- Implement verification pipeline with random prompts – Liam
- Add VAD for live microphone input – Lily
- Integrate scoring and accept/reject decision logic – Leland
- Test live demo on Windows laptop microphone – Leland

## M7: Report and Final Deliverables
- Write methods and system description section – Leland
- Write experiments and results section – Leland
- Write discussion and limitations section – Lily
- Compile ISMIR-style report with 15–20 references – Lily
- Final reproducibility check (scripts + configs) – Liam
