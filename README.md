# Italian Speech Emotion Recognition (SER) Project

This project explores Speech Emotion Recognition (SER) for the Italian language using classical machine learning models. The primary goal is to build and evaluate classifiers capable of distinguishing between five core emotions: **Anger, Joy, Neutral, Sadness, and Fear**.

A key focus of this work is the analysis of model generalization across different audio corpora, a common challenge in audio processing known as **Domain Shift**.

## Table of Contents
1.  [Methodology](#-methodology)
2.  [Experimental Protocol](#-experimental-protocol)
3.  [Results & Analysis](#-results--analysis)
4.  [Understanding the Metrics](#-understanding-the-metrics)
5.  [Conclusion](#-conclusion)

##  Methodology

### Datasets
We utilized two distinct Italian audio corpora, harmonized to a common intersection of labels.

1.  **AI4SER** (Acted/Studio):
    *   A high-quality, acted dataset based on the EMOVO corpus.
    *   **~2,500 files** selected (Filtered out 'Disgust' and 'Surprise').
2.  **EMOZIONALMENTE** (Varied/Spontaneous):
    *   Diverse acoustic environments and speakers.
    *   **~4,900 files** used.
    *   *Data Cleaning:* We identified a labeling inconsistency in the metadata (`neutrality` vs `neutral`). By fixing this mapping, we successfully retrieved the Neutral class samples, solving the initial class imbalance.

**Data Harmonization**: To ensure compatibility, we worked with the intersection of emotion labels present in both datasets (Anger, Joy, Neutral, Sadness, Fear).

### Feature Extraction
We compared two sets of acoustic features to represent the audio signals:

1.  **MFCC (Mel-Frequency Cepstral Coefficients)**: A baseline set of 26 features (13 coefficients + 13 deltas) representing the spectral and timbral qualities of the voice. Extracted using `librosa`.
2.  **eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set)**: A state-of-the-art set of 88 features designed for emotion and paralinguistic analysis. It includes prosodic features like pitch (F0), jitter, shimmer, and loudness, which are crucial for capturing emotional expression. Extracted using `opensmile`.

### Models
We compared two different machine learning models:

1.  **SVM (Support Vector Machine)**: A powerful and robust classical model that works well in high-dimensional feature spaces. We used an RBF kernel.
2.  **MLP (Multi-Layer Perceptron)**: A feed-forward Neural Network with two hidden layers (256 and 128 neurons). This model can capture complex, non-linear relationships between features and scales better with larger datasets.

##  Experimental Protocol

To fully evaluate the models' performance and generalization capabilities, we conducted three main experiments. All intra-dataset and combined scenarios were evaluated using a **5-Fold Stratified Cross-Validation**.

1.  **Scenario 1: Intra-Dataset**:
    -   Train and test on AI4SER.
    -   Train and test on EMOZIONALMENTE.
    -   *Purpose*: To establish a performance baseline for each dataset individually.

2.  **Scenario 2: Cross-Dataset**:
    -   Train on AI4SER, test on EMOZIONALMENTE.
    -   Train on EMOZIONALMENTE, test on AI4SER.
    -   *Purpose*: To measure the models' ability to generalize to unseen acoustic conditions (different microphones, rooms, speakers).

3.  **Scenario 3: Combined**:
    -   Train on a combined dataset of both AI4SER and EMOZIONALMENTE.
    -   *Purpose*: To build a more robust and general-purpose model by exposing it to a wider variety of data.

##  Results & Analysis

The experiments yielded clear and insightful results. Below is the summary of the best performance metrics obtained (using **eGeMAPS** features).

| Scenario | Training Set | Test Set | Best Model | Accuracy | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Intra-Dataset** | AI4SER | AI4SER | SVM | **0.72** | 0.72 |
| **1. Intra-Dataset** | EMOZIONALMENTE | EMOZIONALMENTE | SVM | **0.56** | 0.56 |
| **2. Cross-Dataset** | AI4SER | EMOZIONALMENTE | SVM | **0.29** | 0.27 |
| **2. Cross-Dataset** | EMOZIONALMENTE | AI4SER | SVM | **0.30** | 0.25 |
| **3. Combined** | **AI4SER + EMOZ** | **AI4SER** (Subset) | **MLP** | **0.85** | **0.85** |
| **3. Combined** | **AI4SER + EMOZ** | **EMOZ** (Subset) | **MLP** | **0.74** | **0.74** |

**Key Findings:**
1.  **Feature Superiority**: **eGeMAPS** consistently outperformed MFCCs across all tests, proving that prosodic features are essential for emotion recognition.
2.  **Domain Shift**: The drastic drop in accuracy in Scenario 2 (~30%) confirms that models trained on a single corpus overfit to the recording environment (microphone, room acoustics) rather than learning universal emotion features.
3.  **The Solution**: The **Combined Training** strategy (Scenario 3) using a Neural Network (MLP) was the only effective method to mitigate domain shift, boosting accuracy to **85%** on AI4SER and **74%** on EMOZIONALMENTE.

##  Understanding the Metrics

To evaluate the models, we used the following standard classification metrics:

-   **Accuracy**: The most straightforward metric. It measures the overall percentage of correct predictions.
-   **Precision**: Measures the model's exactness. Of all the times the model predicted a certain emotion, how many times was it right?
-   **Recall (Sensitivity)**: Measures the model's completeness. Of all the actual examples of an emotion, how many did the model correctly identify?
-   **F1-Score**: The harmonic mean of Precision and Recall. It provides a single score that balances both concerns.
-   **Confusion Matrix**: A table that visualizes model performance. The diagonal shows correct predictions, while off-diagonal cells show where the model made mistakes.

##  Conclusion

This project successfully implemented and evaluated a pipeline for Speech Emotion Recognition in Italian.

The key takeaways are:
1.  **Feature selection is critical**: Advanced acoustic features like eGeMAPS are necessary for achieving competitive performance.
2.  **Domain Shift is the primary challenge**: Models trained on a single audio corpus fail to generalize to new acoustic environments.
3.  **Data diversity is the solution**: Combining multiple datasets is the most effective strategy to build robust and general-purpose SER models.