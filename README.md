# Italian Speech Emotion Recognition (SER) Project

This project explores Speech Emotion Recognition (SER) for the Italian language using classical machine learning models. The primary goal is to build and evaluate classifiers capable of distinguishing between five core emotions: **Anger, Joy, Neutral, Sadness, and Fear**.

A key focus of this work is the analysis of model generalization across different audio corpora, a common challenge in audio processing known as **Domain Shift**.

## 📋 Table of Contents
1.  [Methodology](#-methodology)
2.  [Experimental Protocol](#-experimental-protocol)
3.  [How to Run](#-how-to-run)
4.  [Results & Analysis](#-results--analysis)
5.  [Understanding the Metrics](#-understanding-the-metrics)
6.  [Conclusion](#-conclusion)

## 🔬 Methodology

### Datasets
Two distinct Italian SER datasets were used:

1.  **AI4SER**: A high-quality, acted dataset based on the EMOVO corpus. It features multiple speakers reciting sentences with clear emotional intent in a studio environment. The useful subset contains **2,500 audio files**, balanced across the 5 target emotions.
2.  **EMOZIONALMENTE**: A more varied and challenging dataset containing **~4,900 audio files**. The recordings are more spontaneous and exhibit greater acoustic diversity.

**Data Harmonization**: To ensure compatibility, we worked with the intersection of emotion labels present in both datasets. The classes *Disgust* and *Surprise* were filtered out from AI4SER. The `neutrality` label in EMOZIONALMENTE's metadata was mapped to the *Neutral* class.

### Feature Extraction
We compared two sets of acoustic features to represent the audio signals:

1.  **MFCC (Mel-Frequency Cepstral Coefficients)**: A baseline set of 26 features (13 coefficients + 13 deltas) representing the spectral and timbral qualities of the voice. Extracted using `librosa`.
2.  **eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set)**: A state-of-the-art set of 88 features designed for emotion and paralinguistic analysis. It includes prosodic features like pitch (F0), jitter, shimmer, and loudness, which are crucial for capturing emotional expression. Extracted using `opensmile`.

### Models
We compared two different machine learning models:

1.  **SVM (Support Vector Machine)**: A powerful and robust classical model that works well in high-dimensional feature spaces. We used an RBF kernel.
2.  **MLP (Multi-Layer Perceptron)**: A feed-forward Neural Network with two hidden layers (256 and 128 neurons). This model can capture complex, non-linear relationships between features.

## 🧪 Experimental Protocol

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

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Francesco-Mon/Italian-SER-ML-Project.git
    cd Italian-SER-ML-Project
    ```

2.  **Set up the environment:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file from your local environment using `pip freeze > requirements.txt`)*

3.  **Place the data:**
    -   Download the `0000.parquet` and `0001.parquet` files for AI4SER and place them in the project's root directory.
    -   Download `emozionalmente.zip` and place it in the project's root directory.

4.  **Run the notebook:**
    -   Open and run the `Emotion_Recognition_Italian.ipynb` Jupyter Notebook. The cells are designed to be executed in sequence.

## 📊 Results & Analysis

The experiments yielded clear and insightful results:

1.  **eGeMAPS is superior to MFCCs**: Across all scenarios, the eGeMAPS feature set provided a significant performance boost (+5-10% in accuracy) over MFCCs. This confirms that prosodic features are more informative for emotion recognition than purely spectral ones.

2.  **SVM is a strong baseline, MLP shows potential**: SVM provided robust and consistent results. The MLP Neural Network achieved slightly better performance in the "Combined" scenario, indicating its ability to leverage larger, more complex datasets.

3.  **Intra-Dataset performance is high**: On the clean, acted AI4SER dataset, models reached **~75% accuracy**. On the more complex EMOZIONALMENTE dataset, accuracy was around **~60%**, highlighting its challenging nature.

4.  **Cross-Dataset performance reveals a critical weakness**: When training on one dataset and testing on the other, accuracy plummeted to **~30-35%**. This is a classic example of **Domain Shift**, where the model overfits to the acoustic characteristics (microphone, room noise) of the training data and fails to generalize.

5.  **Combined training is the most effective strategy**: By training on a mix of both datasets, the model's accuracy on a cross-validation test reached a robust **~65-70%**. This proves that data diversity is key to building models that can perform well in more realistic scenarios.

## 📈 Understanding the Metrics

To evaluate the models, we used the following standard classification metrics:

-   **Accuracy**: The most straightforward metric. It measures the overall percentage of correct predictions.
    -   *Formula*: `(Correct Predictions) / (Total Predictions)`
    -   *Use Case*: Good for a general overview, but can be misleading on imbalanced datasets.

-   **Precision**: Measures the model's exactness. Of all the times the model predicted a certain emotion, how many times was it right?
    -   *Formula*: `(True Positives) / (True Positives + False Positives)`
    -   *Use Case*: Important when the cost of a false positive is high. For example, you don't want to incorrectly flag a neutral call center conversation as "Anger".

-   **Recall (Sensitivity)**: Measures the model's completeness. Of all the actual examples of an emotion, how many did the model correctly identify?
    -   *Formula*: `(True Positives) / (True Positives + False Negatives)`
    -   *Use Case*: Important when the cost of a false negative is high. For example, you don't want to miss detecting a "Sadness" cue in a mental health monitoring application.

-   **F1-Score**: The harmonic mean of Precision and Recall. It provides a single score that balances both concerns.
    -   *Formula*: `2 * (Precision * Recall) / (Precision + Recall)`
    -   *Use Case*: The best metric for evaluating a model's performance on a per-class basis, especially when classes are imbalanced.

-   **Confusion Matrix**: A table that visualizes model performance. The diagonal shows correct predictions, while off-diagonal cells show where the model made mistakes (e.g., how many times "Sadness" was misclassified as "Neutral").

## ✨ Conclusion

This project successfully implemented and evaluated a pipeline for Speech Emotion Recognition in Italian.

The key takeaways are:
1.  **Feature selection is critical**: Advanced acoustic features like eGeMAPS are necessary for achieving competitive performance.
2.  **Domain Shift is the primary challenge**: Models trained on a single audio corpus fail to generalize to new acoustic environments.
3.  **Data diversity is the solution**: Combining multiple datasets is the most effective strategy to build robust and general-purpose SER models.