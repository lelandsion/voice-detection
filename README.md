# Voice Verification With Noise
Music Information Retrieval (CSC 475)

    Originator: Liam Degand (liamdegand@uvic.ca)
    Collaborators: Leland Sion (lelands@uvic.ca), Lilly Ko (lillyxcko@gmail.com), and Yilun Shi (yilunshi@uvic.ca).


## Project Overview (Abstract)
Voice-based authentication is convenient for device access, but real-world deployment remains difficult when background music and environmental noise are present. This project develops a noise-robust, text-dependent voice verification pipeline for device unlock scenarios at signal-to-noise ratios (SNRs) down to 0 dB. The system combines real-time microphone capture, voice activity detection (VAD), feature extraction, normalization, and verification scoring with random prompt-based evaluation. Current clean-condition baselines on Mozilla Common Voice features show strong separability and competitive initial performance, establishing a foundation for noisy-condition training and evaluation. Our target is to maintain high genuine-user acceptance while keeping impostor acceptance low under adverse acoustic conditions.

## Tools and Datasets
This project uses a Python-based audio pipeline with real-time microphone capture and offline corpus processing. Audio is standardized to 16 kHz mono and processed with framing, windowing, and spectral analysis. Core features include MFCCs, start-frequency descriptors, FFT/DFT spectral summaries, RMS statistics, and zero-crossing characteristics. Normalization and dynamic range conditioning are used to stabilize short-utterance inputs.

For segmentation and robustness, VAD is applied before feature extraction, and noise-aware preprocessing (including filtering and gating) is integrated into the capture and preprocessing flow. The modeling stack currently includes classical baselines (for interpretability) and embedding-oriented directions motivated by modern speaker verification methods.

Training and evaluation primarily use Mozilla Common Voice as the clean speech source, with MUSAN used for controlled noise mixing across SNR levels and VOiCES referenced for realistic reverberant/far-field robustness testing. This combination supports reproducible benchmarking from clean baselines to progressively harder noisy deployment conditions.

## Reproducible Noisy Training Dataset

Use the preprocessing script below to generate a deterministic noisy dataset that mixes each clean utterance with one music clip and one environmental clip at sampled SNR levels.

```bash
python src/noisy_dataset.py \
    --music-noise-dir "data/musan/music" \
    --env-noise-dir "data/musan/noise" \
    --num-augmentations 2 \
    --seed 42
```

Default outputs:

- `data/processed/noisy/audio_files/` (generated noisy wav files)
- `data/processed/commonvoice_noisy.csv` (training metadata with noisy paths and noise details)
- `data/processed/commonvoice_noisy_manifest.json` (full config + generation summary)

To keep results reproducible, keep `--seed`, `--music-snr-db`, `--env-snr-db`, and input datasets fixed.

You can train on the noisy metadata by passing the generated CSV and audio root into existing training loaders (for example, `build_speaker_map` in `src/train.py`).

## Random Prompt Voice Verification Pipeline

The repository now includes a prompt-aware enrollment and verification CLI that:

- captures live microphone audio or uses recorded files,
- extracts speaker embeddings from the trained LSTM encoder,
- compares against enrolled voices using cosine similarity,
- generates random word sequences with `wonderwords` for text-dependent verification.

Install the random-word dependency before running the prompt workflow:

```bash
pip install -r requirements.txt
```

### 1) Enroll a Speaker (live)

```bash
python src/prompt_verification.py enroll \
    --speaker-id alice \
    --model-path checkpoints/ge2e_model.pt \
    --mode live \
    --num-prompts 3 \
    --num-words 3 \
    --takes-per-prompt 2 \
    --seconds 3.0
```

This will generate 3 random prompts of 3 words each and ask the user to record each prompt.

### 2) Enroll a Speaker (recorded files)

```bash
python src/prompt_verification.py enroll \
    --speaker-id alice \
    --model-path checkpoints/ge2e_model.pt \
    --mode recorded \
    --audio-paths data/raw/recordings/alice_1.wav data/raw/recordings/alice_2.wav \
    --prompt-texts "My voice is my password"
```

### 3) Verify with Randomized Prompt

```bash
python src/prompt_verification.py verify \
    --model-path checkpoints/ge2e_model.pt \
    --mode live \
    --claimed-speaker alice \
    --threshold 0.60
```

For recorded verification instead of live microphone input, use `--mode recorded --audio-path <wav_or_mp3_path>`.
Enrollment vectors are saved under `data/processed/enrollment_db/`.

## Related Work
Voice authentication research spans text-dependent and continuous verification settings. Classical statistical systems, especially GMM-based approaches, established early speaker modeling baselines and remain useful reference points due to interpretability. Later work improved short-utterance reliability by incorporating quality-aware scoring under duration limits and noise contamination.

Recent literature has shifted toward robustness in practical environments: continuous/active authentication, replay-aware designs, and stronger front-end conditioning in noisy conditions. In parallel, modern embedding-based methods and end-to-end frameworks have improved discriminative performance by directly learning speaker representations, often coupled with enhancement-aware architectures for low-SNR operation.

Our project follows this trajectory by combining classical baselines with embedding-oriented verification design, VAD-first preprocessing, and realistic noise augmentation/evaluation (Mozilla Common Voice + MUSAN + VOiCES). The emphasis is practical text-dependent unlock verification with random prompts, targeting stable performance in adverse acoustic conditions.

## Objectives, Timeline, Roles


### Train and test model for binary classification of singers based on audio
### Leland

- **PI1 (Basic):** Load and process the datasets including audio.
- **PI2 (Basic):** Train an MFCC-based model for binary classification of singers.
- **PI3 (Expected):** Compare different classifiers in terms of classification accuracy, showing confusion matrices and associated accuracy data.
- **PI4 (Expected):** Implement the model to achieve a desired accuracy of 80–90%.
- **PI5 (Advanced):** Apply data augmentation techniques such as pitch shifting and time stretching, and evaluate how these augmentations impact training performance and classification accuracy.


### Implement the data pipeline for preprocessing audio from raw to model-usable
### Leland

- **PI1 (Basic):** Load raw audio files and convert them into a consistent format for analysis.
- **PI2 (Basic):** Extract relevant audio features (e.g., MFCCs, spectrograms) for model input.
- **PI3 (Expected):** Validate the feature extraction process by visualizing sample spectrograms and checking consistency across files.
- **PI4 (Expected):** Measure processing time for different batch sizes to assess pipeline efficiency.
- **PI5 (Advanced):** Integrate automated error handling and logging for corrupted or missing audio files to improve robustness.


### Implement a conversion from raw mic input into usable raw audio, and create logic/interface for recording audio
### Liam

- **PI1 (Basic):** Capture audio from the microphone and save it in a standard format.
- **PI2 (Basic):** Implement a simple user interface for starting and stopping recordings.
- **PI3 (Expected):** Add real-time monitoring of audio levels and feedback for recording quality.
- **PI4 (Expected):** Implement functionality to store recordings with metadata (timestamp, duration).
- **PI5 (Advanced):** Develop automated preprocessing of recorded audio, including noise reduction and normalization, before sending it to the model.


### Add noise robustness to the model and evaluate performance to a desired level
### Liam

- **PI1 (Basic):** Introduce controlled noise to audio samples to simulate real-world conditions.
- **PI2 (Basic):** Add basic noise isolation to the existing model.
- **PI3 (Expected):** Train the model using noisy inputs to improve robustness.
- **PI4 (Expected):** Implement preprocessing strategies (e.g., filtering, denoising) to mitigate performance loss.
- **PI5 (Advanced):** Apply data augmentation with noise injection and evaluate improvements in model robustness.


### Implement a verification pipeline with random prompts
### Liam

- **PI1 (Basic):** Generate random test prompts and assign them to the model for evaluation.
- **PI2 (Basic):** Capture and log the model’s predictions for verification.
- **PI3 (Expected):** Analyze consistency of model outputs across repeated prompts.
- **PI4 (Expected):** Identify cases where the model fails and categorize errors by type.
- **PI5 (Advanced):** Implement automated feedback to retrain or fine-tune the model based on verification results.


### Perform normalization on raw sound data to be preprocessed and sent to the model
### Lilly

- **PI1 (Basic):** Standardize audio amplitude and sample rate for all input files.
- **PI2 (Basic):** Apply dynamic range compression or scaling to reduce variation across recordings.
- **PI3 (Expected):** Validate normalization by comparing feature distributions before and after processing.
- **PI4 (Expected):** Evaluate the impact of normalization on model input consistency and training stability.
- **PI5 (Advanced):** Automate the normalization process as part of a full preprocessing pipeline for incoming audio streams.


### Compile report and complete formatting for reports
### Lilly

- **PI1 (Basic):** Collect results, figures, and tables from experiments and organize them logically.
- **PI2 (Basic):** Draft the initial report with clear sections for methodology, results, and discussion.
- **PI3 (Expected):** Review report formatting and consistency, including figure captions and references.
- **PI4 (Expected):** Include evaluation metrics, charts, and tables to support conclusions.
- **PI5 (Advanced):** Implement automated scripts to update the report as new results are generated, ensuring reproducibility.

### Feature Extraction, Evaluation, and Baseline Modelling
### Yilun

- **PI1 (Basic):** Preprocess raw audio files into a consistent model-ready format, including start-frequency extraction, normalization, and summary signal statistics.
- **PI2 (Basic):** Convert time-domain audio into frequency-domain representations using FFT/DFT and extract spectral descriptors for downstream analysis and modelling.
- **PI3 (Expected):** Define and implement verification evaluation metrics, including FAR, FRR, EER, ROC, and AUC, and analyze feature separability between same-speaker and different-speaker samples.
- **PI4 (Expected):** Train and evaluate a clean-data baseline model on extracted speech features, reporting benchmark results on the Mozilla Common Voice dataset.
- **PI5 (Advanced):** Extend the baseline to noisy-condition experiments and integrate the evaluation pipeline into the full speaker-verification system for model comparison and threshold tuning.


# Research and References
| Title | URL | Summary | Citation (IEEE) | Sourced by |
|------|-----|---------|----------------|-----------|
| Voice recognition based on adaptive MFCC and deep learning | [link](https://ieeexplore.ieee.org/abstract/document/7603830) | Proposes an enhanced voice recognition method using Adaptive MFCC and deep learning, addressing noise-removal issues that degrade audio quality in existing algorithms. | H.-S. Bae, H.-J. Lee, and S.-G. Lee, “Voice recognition based on adaptive MFCC and deep learning,” in *Proc. 2016 IEEE 11th Conf. on Industrial Electronics and Applications (ICIEA)*, Hefei, China, Jun. 2016, pp. 1542–1546, doi: 10.1109/ICIEA.2016.7603830. | Leland |
| Voiceprint Unlocking Based on MFCC—Exploration of Voiceprint Models Different from Fingerprint | [link](https://ieeexplore.ieee.org/abstract/document/10709042) | Explores smartphone voiceprint unlocking using MFCC-based voice texture recognition, comparing it to fingerprint unlocking and emphasizing personalization and convenience. | L. Zhang et al., "Voiceprint Unlocking Based on MFCC—Exploration of Voiceprint Models Different from Fingerprint," in Proceedings of the 2024 IEEE 2nd International Conference on Image Processing and Computer Applications (ICIPCA), Shenyang, China, 2024, pp. 763-769, doi: 10.1109/ICIPCA61593.2024.10709042. | Leland |
| Voice Activity Detection (VAD) in Noisy Environments | [link](https://arxiv.org/abs/2312.05815)<br>[read](https://arxiv.org/pdf/2312.05815) | Focuses on isolating speech from diverse background noise using a Voice Activity Detection (VAD) system. | J. Ball, “Voice Activity Detection (VAD) in Noisy Environments,” *arXiv*, 2023. https://arxiv.org/abs/2312.05815 | Lilly |
| Quality measures for speaker verification with short utterances | [link](https://www.sciencedirect.com/science/article/abs/pii/S1051200418304287) | Addresses reliable speaker verification using short utterances by combining match and quality scores derived from zero-order Baum–Welch statistics using GMM-UBM. | S. Das, J. Yang, and J. H. L. Hansen, “Quality measures for speaker verification with short utterances,” *Digital Signal Processing*, vol. 88, pp. 66–79, May 2019, doi: https://doi.org/10.1016/j.dsp.2019.01.023. | Lilly |
| Speaker Recognition in Noisy Environments | [link](https://mirkomarras.github.io/dl-voice-noise/) | Describes a deep-learning–based speaker recognition system evaluated under multiple noise types and SNR levels in realistic smart-environment conditions. | M. Marras, “Speaker recognition in noisy environments,” project page, 2019. | Liam |
| Active Voice Authentication | [link](https://www.sciencedirect.com/science/article/abs/pii/S1051200420300178?via%3Dihub) | Explores continuous speaker verification that monitors and validates a user’s voice in real time using very short voice samples. | Z. Meng, M. U. B. Altaf, and B.-H. (Fred) Juang, “Active voice authentication,” *Digital Signal Processing*, vol. 101, p. 102672, Jun. 2020, doi: https://doi.org/10.1016/j.dsp.2020.102672. | Lilly |
| Mozilla Common Voice: An Open Multilingual Speech Corpus for Machine Learning | [link](https://commonvoice.mozilla.org) | Open, community-driven multilingual speech dataset with validated transcripts, designed to support inclusive and accessible ASR research. | Mozilla Common Voice Dataset, Mozilla Foundation, 2020. [Online]. Available: https://commonvoice.mozilla.org. Accessed: Feb. 3, 2026. | Leland |
| AudioUnlock: Device-to-Device Authentication via Acoustic Fingerprints | [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC12609154/) | Presents an acoustic fingerprint–based authentication scheme for smartphones using speakers and microphones as physical identifiers, robust to environmental conditions and attacks. | M. A. Alghamdi et al., “AudioUnlock: Device-to-device authentication via acoustic fingerprints,” *Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.*, vol. 9, no. 3, Oct. 2025. | Liam |
| MUSAN: A Music, Speech, and Noise Corpus | [link](https://arxiv.org/abs/1510.08484) | Open dataset of music, speech, and noise widely used for data augmentation in speaker verification by mixing clean speech with controlled SNR noise. | D. Snyder, G. Chen, and D. Povey, “MUSAN: A music, speech, and noise corpus,” *arXiv preprint* arXiv:1510.08484, 2015. | Liam |
| An unsupervised deep domain adaptation approach for robust speech recognition | [link](https://www.sciencedirect.com/science/article/abs/pii/S0925231217301492) | Applies DNN-HMM systems with Deep Domain Adaptation training to improve speech recognition robustness in noisy environments. | S. Sun, B. Zhang, L. Xie, and Y. Zhang, “An unsupervised deep domain adaptation approach for robust speech recognition,” *Neurocomputing*, vol. 257, pp. 79–87, Sept. 2017, doi: https://doi.org/10.1016/j.neucom.2016.12.063. | Lilly |
| Speaker Verification Using Adapted Gaussian Mixture Models | [link](https://www.sciencedirect.com/science/article/abs/pii/S1051200499903615) | Uses Gaussian Mixture Models (GMMs) to model the statistical characteristics of a speaker’s voice for verification. | B. H. Juang and S. Furui, “Automatic recognition and understanding of spoken language – a first step toward natural human-machine communication,” *Proc. IEEE*, vol. 88, no. 8, pp. 1142–1165, Aug. 2000, doi: https://doi.org/10.1109/5.880077. | Lilly |
| An Overview of Noise-Robust Automatic Speech Recognition | [link](https://ieeexplore.ieee.org/abstract/document/6732927) | Surveys methods for improving ASR robustness to real-world noise and acoustic distortions driven by consumer voice applications. | J. Li, L. Deng, Y. Gong, and R. Haeb-Umbach, “An overview of noise-robust automatic speech recognition,” *IEEE/ACM Trans. Audio, Speech, Lang. Process.*, vol. 22, no. 4, pp. 745–777, Apr. 2014, doi: 10.1109/TASLP.2014.2304637. | Leland |
| Voice Identification in Python Using Hidden Markov Model | [link](https://www.researchgate.net/profile/Harika-Kotha/publication/342956180_Voice_Identification_in_Python_Using_Hidden_Markov_Model/links/5f0eff2b92851c1eff11e854/Voice-Identification-in-Python-Using-Hidden-Markov-Model.pdf) | Describes a Python-based voice identification system using Hidden Markov Models, highlighting NLP-driven speech recognition applications. | V. M. N. S. V. K. Gupta, R. Shiva Shankar, H. D. Kotha, and J. Raghaveni, “Voice identification in Python using Hidden Markov Model,” *Int. J. Adv. Sci. Technol.*, vol. 29, no. 6, pp. 8100–8112, 2020. | Leland |
| VOiCES: Voices Obscured in Complex Environmental Settings (VOiCES Corpus) | [link](https://www.isca-archive.org/interspeech_2018/richey18_interspeech.html) | Corpus of 15,904 speech segments from 196 speakers recorded in real rooms with reverberation, multiple microphones, and background noise for far-field ASR and speaker recognition. | C. Richey et al., “Voices Obscured in Complex Environmental Settings (VOiCES) corpus,” in *Proc. Interspeech*, 2018, pp. 1566–1570. | Liam |
| Extended U-Net for Speaker Verification in Noisy Environments | [link](https://www.isca-archive.org/interspeech_2022/kim22b_interspeech.pdf) | Proposes U-Net and Extended U-Net architectures that jointly perform enhancement and speaker embedding extraction, achieving state-of-the-art results on VoxCeleb1 mixed with MUSAN and VOiCES. | J.-h. Kim, J. Heo, H.-j. Shim, and H.-J. Yu, “Extended U-Net for speaker verification in noisy environments,” in *Proc. Interspeech*, 2022, pp. 590–594. | Liam |
| VoiceLive: A Phoneme Localization based Liveness Detection for Voice Authentication on Smartphones | [link](https://dl.acm.org/doi/abs/10.1145/2976749.2978296) | Uses smartphone stereo microphones for liveness-aware voice authentication resistant to replay attacks. | L. Zhang, S. Tan, J. Yang, and Y. Chen, “VoiceLive: A phoneme localization based liveness detection for voice authentication on smartphones,” in *Proc. 2016 ACM SIGSAC Conf. on Computer and Communications Security (CCS)*, Vienna, Austria, 2016, pp. 1080–1091, doi: 10.1145/2976749.2978296. | Lilly |

## Progress Report Update (March 2026)

### Initial Results Summary

- Built an end-to-end preprocessing and verification foundation with standardized 16 kHz mono microphone capture, recording metadata logs, normalization, and feature extraction.
- Feature pipeline currently supports MFCC, start-frequency, FFT/DFT-based descriptors, RMS, and zero-crossing-derived statistics.
- Prepared **3,951 usable clips** from **418 speakers** from Mozilla Common Voice en-AU for analysis.
- Feature separability subset used **3,390 utterances** from **152 speakers** and showed:
    - same-speaker cosine similarity: **0.6802 ± 0.2254**
    - different-speaker cosine similarity: **0.0446 ± 0.2753**
- Clean-data baseline (logistic regression) reached:
    - **~91.8% mean validation accuracy**
    - **~98.8% top-5 accuracy**

### Progress By Contributor

#### Liam (Mic Input, Noise Robustness, Verification Pipeline)

- Completed microphone pipeline milestones for capture and interface:
    - standardized microphone recording to 16 kHz mono WAV
    - start/stop recording workflow
    - real-time RMS monitoring
    - metadata logging in `data/processed/recordings_metadata.csv`
    - automated preprocessing hooks (normalization and noise gating)
- Completed early noise-robustness work:
    - controlled noisy preprocessing and filtering foundations
    - VAD integrated in the pipeline flow
- Remaining work:
    - train and evaluate full robustness under noisy conditions (PI3-PI5 for noise robustness)
    - complete verification pipeline milestones with random prompts (remaining expected/advanced indicators)

#### Leland (Modeling + Data Pipeline)

- Completed repository/project setup milestones (structure, timeline, data workflow planning).
- Completed end-to-end data extraction and train/test preparation pipeline.
- Implemented baseline training on filtered clean data with reproducible evaluation.
- In progress:
    - model checkpointing and configuration management
    - robustness evaluation across multiple noise conditions

#### Lilly (Normalization + Report Integration)

- Completed core normalization objectives:
    - sample-rate standardization to 16,000 Hz
    - RMS-targeted amplitude normalization (targeting stable loudness)
    - dynamic range compression to reduce frame-to-frame level variance
    - distribution comparison tools to validate normalization effects
- Completed normalization impact analysis toward model-input consistency.
- Ongoing:
    - integration of normalization improvements with training stability analysis and report updates

#### Yilun (Feature Extraction, Metrics, Baseline Evaluation)

- Completed PI1 and PI2:
    - preprocessing scripts from raw recordings to structured model features
    - frequency-domain extraction (FFT/DFT summaries including dominant-frequency-related descriptors)
- Completed PI3:
    - reusable verification-metrics tooling (FAR, FRR, EER, ROC, AUC) with reproducible JSON/CSV outputs
    - feature-separability analysis scripts for same- vs different-speaker comparisons
- Completed PI4:
    - clean-data baseline training and benchmarking on Mozilla Common Voice feature sets
- Next step (PI5):
    - extend baseline and evaluation to noisy-condition experiments and integrate into final verification comparisons

### Current Development Focus

- Model checkpointing and reproducible experiment configuration.
- Noise-condition evaluation at varying SNR levels and background types.
- Extending verification experiments from clean baseline to robust noisy settings.
- Final integration of metrics and reporting for the ISMIR-style deliverable.

## Future Work

### Basic Goal Completion

- Maintain and polish a complete prototype that ingests speech audio, extracts core features, and verifies speaker identity against enrolled data.

### Expected Goal Completion

- Complete a full noise-robust verification pipeline with:
    - robust enrollment flow
    - random-prompt verification workflow
    - expanded VAD integration for live microphone usage
    - systematic evaluation under controlled low-SNR noise conditions

### Advanced Extensions

- Move beyond the current clean baseline by training and validating on noisy-condition datasets.
- Compare additional speaker-embedding architectures (for example x-vector and ECAPA-TDNN style systems) against current baselines.
- Expand evaluation to more realistic deployment conditions, including reverberation, far-field microphones, and mixed environmental noise.
- Add security-oriented checks such as replay-attack resilience, liveness verification, and continuous authentication behavior.

## Bibliography

1. H.-S. Bae, H.-J. Lee, and S.-G. Lee, “Voice recognition based on adaptive MFCC and deep learning,” in Proceedings of the 2016 IEEE 11th Conference on Industrial Electronics and Applications (ICIEA), Hefei, China, Jun. 2016, pp. 1542–1546, doi: 10.1109/ICIEA.2016.7603830.

2. J. Ball, “Voice Activity Detection (VAD) in Noisy Environments,” arXiv, 2023. [Online]. Available: https://arxiv.org/abs/2312.05815

3.  S. Das, J. Yang, and J. H. L. Hansen, “Quality measures for speaker verification with short utterances,” Digital Signal Processing, vol. 88, pp. 66–79, May 2019, doi: 10.1016/j.dsp.2019.01.023.

4. M. Marras, “Speaker recognition in noisy environments,” project page, 2019. [Online]. Available: https://mirkomarras.github.io/dl-voice-noise/

5. Z. Meng, M. U. B. Altaf, and B.-H. Juang, “Active voice authentication,” Digital Signal Processing, vol. 101, p. 102672, Jun. 2020, doi: 10.1016/j.dsp.2020.102672.

6. Mozilla Foundation, “Mozilla Common Voice: An Open Multilingual Speech Corpus for Machine Learning,” 2020. [Online]. Available: https://commonvoice.mozilla.org. Accessed: Feb. 3, 2026.

7. M. A. Alghamdi et al., “AudioUnlock: Device-to-device authentication via acoustic fingerprints,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 9, no. 3, Oct. 2025.

8. D. Snyder, G. Chen, and D. Povey, “MUSAN: A music, speech, and noise corpus,” arXiv preprint arXiv:1510.08484, 2015.

9. S. Sun, B. Zhang, L. Xie, and Y. Zhang, “An unsupervised deep domain adaptation approach for robust speech recognition,” Neurocomputing, vol. 257, pp. 79–87, Sept. 2017, doi: 10.1016/j.neucom.2016.12.063.

10. B. H. Juang and S. Furui, “Automatic recognition and understanding of spoken language – a first step toward natural human-machine communication,” Proceedings of the IEEE, vol. 88, no. 8, pp. 1142–1165, Aug. 2000, doi: 10.1109/5.880077.

11. J. Li, L. Deng, Y. Gong, and R. Haeb-Umbach, “An overview of noise-robust automatic speech recognition,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 4, pp. 745–777, Apr. 2014, doi: 10.1109/TASLP.2014.2304637.

12. V. M. N. S. V. K. Gupta, R. Shiva Shankar, H. D. Kotha, and J. Raghaveni, “Voice identification in Python using Hidden Markov Model,” International Journal of Advanced Science and Technology, vol. 29, no. 6, pp. 8100–8112, 2020.

13. C. Richey et al., “Voices Obscured in Complex Environmental Settings (VOiCES) corpus,” in Proceedings of Interspeech, 2018, pp. 1566–1570.

14. J.-H. Kim, J. Heo, H.-J. Shim, and H.-J. Yu, “Extended U-Net for speaker verification in noisy environments,” in Proceedings of Interspeech, 2022, pp. 590–594.

15. L. Zhang, S. Tan, J. Yang, and Y. Chen, “VoiceLive: A phoneme localization based liveness detection for voice authentication on smartphones,” in Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS), Vienna, Austria, 2016, pp. 1080–1091, doi: 10.1145/2976749.2978296.

16. L. Zhang et al., "Voiceprint Unlocking Based on MFCC—Exploration of Voiceprint Models Different from Fingerprint," in Proceedings of the 2024 IEEE 2nd International Conference on Image Processing and Computer Applications (ICIPCA), Shenyang, China, 2024, pp. 763-769, doi: 10.1109/ICIPCA61593.2024.10709042.
