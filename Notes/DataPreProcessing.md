---
title: Data PreProcessing
nav_order: 1
parent: Notes
layout: default
---

# Data Pre-Processing

Necessary for accuracy in analysis. Real world data is often imperfect or inconsistent.
<hr>

## Common Problems in Data

1. ### Data Corruption and Noise
    - Reasons include:
        - Sensor Issues
        - Measurement Process Issues
2. ### Irrelevant Data
    - Raw Data often contains extremely large number of features and samples
    - Most often irrelevant
    - Feature selection more advanced process to choose features
    - Sample selection techniques find more relevant samples
3. ### Fractured Data
    - Same dataset can have fractures due to various factors:
        1. Incompatible data: Can be due to differences in sensors/measurement mediums
        2. Multiple Sources: Data from multiple sources might require converting to same format
        3. Data from multiple levels of granularity: Different sensors can have different levels of precision
4. ### Missing attributes
    - Samples and features can be missing for all or for a segment of data

## Common Data Pre-Processing

1. ### Data Cleaning
   Required when data:
    1. Is in inconsistent format (e.g., dates like dd-mm-yy and mm-dd-yyyy together)
    2. Has duplicates
    3. Has missing values
    4. Is noisy

2. ### Handling Missing Data
   Techniques:
    1. Remove samples (only if classes are balanced and data is not limited)
    2. For numeric data: Replace with dataset statistical metrics for the entire dataset or per class (e.g., mean, mode,
       median)
    3. For categorical data: Replace with mode or "NA"
    4. For time series: Interpolate (e.g., linear interpolation between known points)
    5. Regress using other variables (e.g., use regression models to predict missing values based on correlated
       features)

3. ### Outliers
    - Data points that are statistical anomalies
    - Significantly deviate from other observations in the dataset
    - Can arise due to measurement errors, natural variation, or rare events
    - Can be valid or invalid observations

4. ### Outlier Detection
    - ### Basic Methods:
        1. ### Z-Score:
            - Z-score is calculated as $$z = \frac{x - \mu}{\sigma}$$.
            - Usually safe to consider 𑁇z𑁇 > 3 as outliers (based on the Central Limit Theorem).
        2. ### IQR (Interquartile Range):
            - IQR = Q3 - Q1 (where Q1 is 25th percentile, Q3 is 75th percentile).
            - Calculated using median. Common to consider only points within
                - [**Q1** - 1.5 × IQR, **Q3** + 1.5 × IQR]
        3. ### Mahalanobis Distance:
            - Measures the distance of a point from the mean, accounting for the covariance among variables. Useful for
              multivariate outliers.
            - Steps:
              1. Transforms variables into uncorrelated variables
              2. Scale to make variance equal to 1
              3. Calculate euclidean distance
            - Why? 
              - If variables are correlated, as A increases so will B
              - Points logically closer will have extra correlated distance added to them
        4. ### Box Plot Analysis:
            - Visualization method based on IQR to identify outliers.
    - ### ML Algorithms Based:
        1. ### DBSCAN:
            - Density based spatial clustering with noise
            - Finds outliers automatically
        2. ### Auto Encoders:
            - High reconstruction error can be used as a parameter to identify outliers

5. ### Handling Outliers
   When and How?
    1. If due to measurement error or inconsistency: Remove or replace
        1. Replace using imputation techniques (e.g., mean/median replacement or model-based imputation)
    2. If rare but valid: Keep, as they usually contain useful signal
    3. While scaling: Use robust scaling to minimize impact

6. ### Data Transformation (Scaling)
    1. ### Standardization:
        - (mean = 0, variance = 1)**:
        - Replace values with Z-score
        - $$z = \frac{x - \mu}{\sigma}$$
    2. ### Min-Max Scaling (Normalization): Scale to [0, 1]
        - $$x_i' = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}$$
    3. ### Robust Scaling:
        - Scaling using Q3 and Q1, where IQR = Q3 - Q1
        - $$x_i' = \frac{x_i - \median(X)}{\IQR(X)}$$
        - ℹ️ Note: 90th percentile means 90% data lies below this
        - ℹ️ Note: Q3 and Q1 are 75th and 25th percentile respectively
    4. ### Log Transform:
        - Used when values are extremely large to bring them to a comparable range
        - $$x_i' = \log(x_i)$$
        - Handles outliers well, even if not limiting all values to [0, 1]
    5. ### **Box-Cox Transform**:
        - A power transformation to stabilize variance and make the data more normally distributed.
        - $$x_i' = \begin{cases}
          \frac{x_i^\lambda - 1}{\lambda} & \lambda \neq 0 \\
          \log(x_i) & \lambda = 0
          \end{cases}$$

7. ### Encoding Categorical Variables
    1. **One-Hot Encoding**: Replace categories with 1 in their index, 0 in others
        - 🟢 Maintains independence of classes
        - 🔴Grows linearly with number of classes (not suited for high cardinality)
    2. **Label Encoding**: Assign integers to each class
        - 🔴Integers assigned can cause the model to assume numerical relationships where none exist
    3. **Ordinal Encoding**: Same as label encoding, but used when order actually exists (e.g., Small, Medium, Large)
    4. **Binary Encoding**: Similar to one-hot but more space-efficient and less independent (converts classes to binary
       representations, reusing dimensions but potentially mixing information)
    5. **Hash Encoding**: Maps high-cardinality categories to fixed-size vectors using a hash function (collisions can
       occur)
    6. **Target Encoding**: Replace category with the mean of the target variable for samples in that category
        - 🔴Can cause leakage of target information into features
    7. **Frequency Encoding**: Replace classes with their frequency of occurrence
        - 🟢 Can simplify encoding for very high cardinality
        - 🔴Maps multiple categories to the same value and can be confusing when no inherent imbalance

## Dimensionality Reduction Techniques

Reduce the number of features in a dataset while preserving information.  
Done via:

1. **Feature Selection**: Selecting a subset of features
2. **Feature Extraction**: Creating new features that extract all important information from the data

## Feature Selection

1. ### Variance Thresholding
   If a feature shows low variance, it has a high probability of not contributing to prediction.  
   Works only if low variance is meaningless (e.g., near-constant features), but variance can be useful in certain
   cases.

2. ### Correlation Filtering
    - Correlated variables can cause multicollinearity.
    - **Multicollinearity**: When features are linearly dependent.
        - **Issues**: Destabilizes training as the model can't distinguish which feature to assign weights to;
          expressive power becomes shared (e.g., coefficients could be 1:9, 1:1, or -2:12).
    - Techniques:

    1. Pearson Correlation (for numeric-numeric data):
       $$r_{X,Y} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$
       i.e., $$\cov(X,Y) / (\std(X) \std(Y))$$
    2. Correlation Matrix: Calculated using Pearson/Spearman/Kendall. The matrix compares every feature pair and looks
       for high correlations.
    3. Chi-Square Test: For categorical features
    4. Monotonic Increase (Spearman): For non-linear correlations (e.g., height and weight in a population)
       $$\rho_{X,Y} = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}$$
       where $d_i = R(x_i) - R(y_i)$ and $R$ is the rank (position in sorted ascending order)
    5. Kendall: Measures ordinal association
    6. Mutual Information Score: Uses KL Divergence
        - KL Divergence: The difference between two probability distributions in terms of bits (extra bits required to
          encode one distribution using the other).
          $$I(X;Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log \left( \frac{p(x,y)}{p(x)p(y)} \right)$$
            - ℹ️ If $p(x,y) = p(x)p(y)$, they are independent (log term becomes 0, no mutual information).

## Feature Extraction

Techniques:

1. **PCA (Principal Component Analysis)**: Linear technique that transforms data into uncorrelated components ordered by
   variance explained.[PCA](MachineLearning.md#PCA)
2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Non-linear dimensionality reduction for visualization,
   preserves local structure.[T-SNE](MachineLearning.md#T-SNE)
3. **UMAP (Uniform Manifold Approximation and Projection)**: Similar to t-SNE but faster and better at preserving global
   structure.[UMAP](MachineLearning.md#UMAP)

## Sampling

Process of selecting a subset from a larger group (population) to make inferences about the whole (too costly and
inefficient to test the entire population).

1. **Parameter vs Statistic**
    - Parameter: True numeric value describing the population
    - Statistic: Numeric value derived from the sample

2. **Probability Sampling**
    - Every member of the population has a non-zero chance of being selected.

    1. **Simple Random Sampling**
        - Every individual has an equal chance of being selected.
        - 🟢 Pros: Unbiased if truly random and easy to analyze
        - 🔴Cons: Impractical due to requiring a full list of the population
    2. **Stratified Sampling**
        - Divide population into homogeneous subgroups (strata) and take random samples from each (e.g., divide by age).
        - 🟢 Pros: Ensures representation of all groups
        - 🔴Cons: Requires complex design and knowledge to divide strata
    3. **Systematic Sampling**
        - Select every k-th item.
        - 🟢 Pros: Easy, good coverage
        - 🔴Cons: Hidden numerical patterns can cause bias; requires full list
    4. **Cluster Sampling**
        - Divide population into clusters and select one or more entire clusters.
        - 🟢 Pros: Cheaper and convenient
        - 🔴Cons: Less precise than stratified; bias due to only some clusters selected
    5. **Multi-Stage Sampling**
        - Combines several sampling methods (e.g., cluster then random within clusters).

3. **Non-Probability Sampling**
    - Individuals have either unknown or unequal chances of being selected.

    1. **Convenience Sampling**
        - Sample whoever is accessible.
        - 🟢 Pros: Cheap and convenient
        - 🔴Cons: Extremely biased
    2. **Purposive/Judgmental Sampling**
        - Researcher selects samples based on expert judgment or specific characteristics.
        - 🟢 Pros: Targeted and efficient for qualitative research
        - 🔴Cons: Subjective and prone to bias
    3. **Quota Sampling**
        - Like stratified, but non-random selection within predefined quotas (e.g., interview 50 men and 50 women).
        - 🟢 Pros: Ensures diversity without full population list
        - 🔴Cons: Non-random, potential selection bias
    4. **Snowball Sampling**
        - Start with initial subjects who refer others (useful for hard-to-reach populations, e.g., rare disease
          patients).
        - 🟢 Pros: Effective for hidden populations
        - 🔴Cons: Biased toward connected individuals; hard to control
    5. **Sampling Distributions**
        - The distribution of a statistic (e.g., sample mean) over many samples. Central Limit Theorem: Sample means
          approximate normal distribution for large n, regardless of population shape.

4. **Over- and Under-Sampling**: Handling class imbalance by oversampling the minority class or undersampling the
   majority class to create balance.
   Techniques:
    1. ### SMOTE (Synthetic Minority Oversampling Technique)
        - For the minority class, use k-nearest neighbors to create synthetic samples by interpolating between
          neighbors.
        - 🔴Not real data, so might cause overfitting
    2. ### ADASYN (Adaptive Synthetic Sampling)
        - Similar to SMOTE but focuses on generating more samples near decision boundaries (minority points with many
          majority neighbors).
        - 🟢 Teaches the model more by focusing on difficult regions
        - 🔴Can introduce noise
    3. ### Undersampling
        - Randomly remove samples from the majority class (e.g., RandomUnderSampler or NearMiss for intelligent
          selection).
    4. ### Class Weight Adjustment
        - Assign higher weights to minority class during model training to penalize misclassifications more.

## Text Preprocessing

1. ### Lowercasing
    - Standardize text to lowercase for consistency.

2. ### Tokenization
    - Splits text into smaller units (tokens), usually words.
    - Types:
        - Word tokenization: Splits into individual words.
        - Character tokenization: Splits into characters.
            - 🟢 Small vocabulary
            - 🔴Computationally intensive; doesn't capture context
        - Sentence tokenization: Splits into sentences (low use case).
        - Subword tokenization: Breaks into smaller units than words (useful in deep learning).  
          Two popular types: (See [Modern Tokenizations](LLM.md#Modern-tokenization))
            1. Byte Pair Encoding (BPE)
            2. WordPiece Encoding

3. ### Stopword Removal
    - Removes common words that appear frequently but add little meaning (e.g., "the", "is", "and").
    - 🟢 Reduces dimensionality and focuses on meaningful words.

4. ### Stemming and Lemmatization
    - Stemming: Reduces words to their base form (may not be a real word).
        - Example: "running" → "run", "flies" → "fli"
    - Lemmatization: Converts to meaningful base form using vocabulary and grammar.
        - Example: "running" → "run", "better" → "good"
    - Lemmatization is usually more accurate than stemming.

5. ### Others:
    - Removing punctuation, numbers, and special characters
    - Removing duplicates and blank lines
    - Spelling correction
    - Slang and abbreviation handling (optional)
    - Expanding contractions (e.g., "can't" → "cannot")
    - HTML tag removal (for web-scraped text)
    - Emoji handling (remove or convert to text)
    - Ensures consistency across the dataset.

6. ### Vectorization
    - Converts processed text into numerical format.
    - Common methods:
        1. Bag of Words (BoW): Counts word frequency in each document.
        2. TF-IDF (Term Frequency-Inverse Document Frequency): Weighs words by importance across documents.
        3. Word Embeddings: Represents words in dense vector form based on meaning (e.g., Word2Vec, GloVe, BERT).

## Word Embedding

- Goal: Convert words into vectors such that similar words (in meaning/context) are close in embedding space (measured
  by cosine similarity or Euclidean distance).
- Types:
    1. ## Static Embeddings [DEPRECATED]
        - Created with a large but finite corpus; only works with words in vocabulary.
        - 🔴Doesn't handle polysemy well (words with multiple meanings).  
          Notable Implementations:

        1. **Word2Vec**: Trained using a neural network with input (W_in) and output (W_out) embeddings.
            1. CBOW (Continuous Bag of Words): Predicts middle word from surrounding words (averaged embeddings).
            2. Skip-Gram: Predicts surrounding words from middle word (better for rare words).
        2. **GloVe (Global Vectors for Word Representation)**: Uses global co-occurrence matrix (words on axes,
           co-occurrence counts in cells).  
           Trained to satisfy
            - $$w_i^T \tilde{w}_j + b_i + \tilde{b}_j \approx \log(X_{ij})$$
            - 🟢 Captures linear relationships (e.g., king - man + woman ≈ queen).
        3. **FastText**: Extension of Word2Vec that uses subword information (n-grams) to handle OOV words and
           morphology.

    2. ## Dynamic Embeddings
        - Embeddings change based on context; used by modern language models.
        - | Model | How it works |
                                        |-------|--------------|
          | ELMo | Uses deep biLSTM to generate embeddings from the entire sentence. |
          | BERT | Uses transformers and attention for bidirectional context-aware embeddings. |
          | GPT  | Uses transformer decoders for left-to-right context; produces dynamic embeddings during generation or fine-tuning. |

## Word Embedding in Non-Textual Context

- **Item2Vec**: Generates dense vector representations of items (e.g., songs, movies, books) based on sequences (similar
  to Word2Vec).
- **Wav2Vec**: Converts audio waveforms into vector representations (self-supervised learning on speech data).