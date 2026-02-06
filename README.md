# Voice Verification With Noise
Music Information Retrieval (CSC 475)

    Originator: Liam Degand (liamdegand@uvic.ca)
    Collaborators: Leland Sion (lelands@uvic.ca), Lilly Ko (lillyxcko@gmail.com)


## Project Overview (Abstract)
Voice-based authentication has emerged as a convenient and secure method for device access control. However, real-world deployment scenarios often involve challenging acoustic conditions, including background music and environmental noise, which significantly degrade verification performance. This paper presents a noise-robust voice verification system designed for device unlock authentication under signal-to-noise ratios (SNRs) as low as 0 dB. The system employs text-dependent speaker verification using random word prompts to authenticate enrolled users while defending against casual imposter attempts. Our target performance criteria are greater than 90% genuine acceptance rate with less than 5% impostor acceptance rate at 0 dB SNR. This project culminates in a live demonstration system that validates the approach using real-time microphone input to detect speech, extract voice features, and authenticate users.

## Tools and Datasets
[]

## Related Work
[]

## Objectives, Timeline, Roles
Found at [PROJECT-STRUCTURE.md](https://github.com/lelandsion/voice-detection/blob/main/PROJECT_STRUCTURE.md)

## Example George Tzanetakis

     Objective (implement a functional genre classification system for the Music4All dataset) - 2-3 objectives per person
        PI1 (basic): load and process the Music4All dataset including audio 
        PI2 (basic): train a Naive Bayes classifier for genre prediction 
        PI3 (expected): compare different classifiers in terms of classification accuracy, show corresponding confusion matrices 
        PI4 (expected): calculated timing stats for training and prediction for each classifier, run different subsets of the dataset to get a better sense of scaling 
        PI5 (advanced) - implement data augmentation by pitch shifting/time stretching and see how it affects training and prediction 


## Research and References
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
