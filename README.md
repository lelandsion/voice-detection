# Voice Verification With Noise

In the presence of music or common environmental noise create a software which is able to recognize a given user's voice to unlock a device. The device should only be unlocked given the registered user's voice which will be analyzed by prompting the user to repeat a set of random words to be analyzed. 

## Project Focus
- Build a centralized classification pipeline for audio detection and isolation of users voice.
- Target use case: User device verification using audio as an input. 
- Tech stack: 

## Datasets and Research
- https://www.isca-archive.org/interspeech_2020/chowdhury20b_interspeech.html
- https://zenodo.org/records/1442513
- https://arxiv.org/abs/2406.04140
- https://defined.ai/datasets/vocal-music-tracks
- https://ieeexplore.ieee.org/abstract/document/6732927
- https://ieeexplore.ieee.org/abstract/document/7603830
- https://www.researchgate.net/profile/Harika-Kotha/publication/342956180_Voice_Identification_in_Python_Using_Hidden_Markov_Model/links/5f0eff2b92851c1eff11e854/Voice-Identification-in-Python-Using-Hidden-Markov-Model.pdf

## Project Plan (current)
- MVP target: >90% genuine-accept and <5% impostor-accept at SNRs down to ~0 dB on a small validation set.
- Scope: single-user enrollment with a few enrollment utterances; evaluation focused on verification (yes/no) not transcription.

## Project Board / Milestones / Issues
- See `PROJECT_STRUCTURE.md` for the GitHub Project, milestones, issues, labels, and roles for **Voice Verification With Noise (Live Demo)**. The project board should be assigned only to the requester and Leland; individual issues remain unassigned by default.

## Pipeline Outline
- Data prep: curate clean speech per enrolled user; mix with music/noise at varied SNRs; hold out speakers for impostor trials.
- Feature extraction: log-mel spectrograms + optional voice activity detection; experiment with MFCC as baseline.
- Modeling: start with x-vector/ECAPA-TDNN embeddings + cosine/PLDA scoring; compare with a compact CNN or wav2vec-style frozen encoder + linear head.
- Evaluation: equal error rate (EER), minDCF, ROC/DET curves across SNR bands; measure latency on CPU-only path.
- Serving sketch: lightweight inference script that records phrase, performs VAD, embeds, and scores against enrolled template; refuse unlock on low confidence or too-short audio.

## Baseline Checklist
- Implement data mixer that layers speech with music/noise at randomized SNRs and start times.
- Train a simple MFCC + cosine baseline for a quick ceiling/floor.
- Implement x-vector or ECAPA-TDNN baseline with pretrained weights where possible; fine-tune on mixed data.
- Add evaluation harness for EER/minDCF and noise-stratified metrics; plot ROC/DET.
- Package a CLI demo (enroll, verify) for the enrolled user.

## Resources (links)
- Overleaf project: https://www.overleaf.com/project/69790f5fbd108f1a4df5de62
- ISMIR 2025 templates: https://www.overleaf.com/latex/templates/paper-template-for-ismir-2025/xnqdhjhdgsfd
- Recent ISMIR submission: https://ismir2025program.ismir.net/papers.html?filter=keywords

## Contacts
- Originator: Liam Degand (liamdegand@uvic.ca)
- Potential collaborators: Leland Sion (lelands@uvic.ca), Lilly Ko (lillyxcko@gmail.com) 
