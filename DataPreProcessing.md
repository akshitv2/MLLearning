# Data Pre-Processing

## Common Data Pre-Processing

1. ### Data Cleaning
   Required when data:
    1. In inconsistent format e.g. dd-mm-yy mm-dd-yyyy together
    2. Has Duplicates
    3. Missing values
    4. Noisy

2. ### Handling Missing Data
   Techniques:
    1. Remove samples (only if classes are balanced and data is not limited)
    2. For numeric: Replace with population metric for entire DS or class e.g. Mean, Mode, Median
    3. For Categorical: Replace with mode or NA
    4. For time series: interpolate
    5. Regress using other variables

3. ### Outlier Detection
   Methods:
    1. **Z Score**: z score is calculated as ((x - mean) / std dev). Usually safe to ignore |z|>3 (Central limit
       theorem)
    2. **IQR (Inter Quartile Range)**: Calculated using median, Common to only consider points in Q3-Q1 (75th to 25th
       Percentile)
    3. Mahalanobis Distance:
    4. **Box Plot Analysis**:

4. ### Data Transformation (Scaling)
    1. **Standardization: (mean = 0, variance = 1)** -> Replace values with Z score
       $$z = \frac{x - \mu}{\sigma}$$
    2. **Min Max Scaling (Normalization)**: Scale 0 to 1
       $$x_i' = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}$$
    3. **Robust Scaling**: Min max scaling except uses Q3 and Q1 ()
       $$x_i' = \frac{x_i - \text{Median}(X)}{\text{IQR}(X)}$$
    4. **Log Transform**: Used when values are extremely large to bring to comparable range
       $$x_i' = \log(x_i)$$
       🟢Handles outliers well even if not limited all values to 0 to 1
    5. Box Core Transform:

5. ### Handling Outliers
   When and How?
    1. If Measurement Error/Inconsistent -> Remove or replace
        1. Replace using imputation techniques
    2. If rare but valid -> Keep since usually contains signal
    3. While scaling using Robust scaling

6. ### Encoding Categorical Variables
    1. **One hot Encoding**: Replace categories with one in their index, 0 in others
        - 🟢 Mantains independence of classes
        - 🔴 Grows Exponentially to classes (not suited for high cardinality)
    2. **Label Encoding**: Assign int to each class
        - 🔴Integer assigned can cause model to assume numerical relationship where none exists
    3. **Ordinal Encoding**: Same as Label except order actually exists like Small Medium Large
    4. **Binary Encoding**: Similar to one hot but more space efficient and less independent
    5. **Hash Encoding**: Similar to binary, meant to map high cardinality to individual spaces
    6. **Target Encoding**: Replace category entirely with mean of target (of samples with said class)
        - 🔴 Can cause leakage of target into sample
    7. **Frequency Encoding**: Replace classes with frequency of entries
        - 🟢 Can simplify encoding of very high cardinality
        - 🔴 Maps multiple categories to same and can be confusing when no imbalance

## Dimensionality Reduction Techniques

Reduce number of features in dataset while preserving info.  
Done via:

1. Feature Selection: Selecting subset of features
2. Feature Extraction: Create new features extracting all important info from data

## **Feature Selection**

1. ### Variance Thresholding
   If feature shows low variance -> High Probablity not contributing to prediction
2. ### Correlation Filtering
   Correlated variables can cause multicollinearity.
   **MultiCollinearity** when features are linearly dependent.  
   Issues? Destabilizes training as model can't understand which feature to increase weights for.
   Expressive power becomes shared (can 1:9 or 1:1 or -2:12).
   Techniques:
    1. Pearson Correlation (for numeric <-> numeric data):
       $$r_{X,Y} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \, \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$
       i.e Cov(x,y)/std(x)std(y)
    2. Correlation Matrix: Calculated using Pearson. Matrix puts every feature X <-> X and looks for higher correlation.
    3. ChiSquare Test for Categorical Features
    4. Monotonic Increase (Spearman): i.e non linear correlated increase for e.g height and weight in population
       $$\rho_{X,Y} = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}$$
       where d<sub>i</sub> = R(x<sub>i</sub>) - R(y<sub>i</sub>) where R is the rank
       ℹ️ Note: Rank here is the position in data if sorted in ascending order
    5. Kendall
    6. Mutual Information Score: Uses KL Divergence
       ℹ️ Note: KL Divergence is the difference between two probablity distribution in terms of bits. Best understood as
       extra bits required to encode from one dist to another.
       $$I(X;Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \, \log \left( \frac{p(x,y)}{p(x)\,p(y)} \right)$$
       Here if p(x,y) = p(x)p(y) then they are independent and log of this becomes 0 i.e. no mutual info between them.

## Feature Extraction

Techniques:

1. PCA
2. T-SNE
3. UMAP

## **Sampling**

Process of selecting a subset from a larger group (called population) to make inferences about the whole.
(Too costly and inefficient to test whole)

1. Parameter vs Statistic

- Parameter: True numeric value describing the population
- Statistic: Numeric value derived from the sample.

1. Probability Sampling
    - Every member of population has a non-zero chance of being selected.

    1. Simple Random Sampling
        - Every individual has equal chance of being selected
        - 🟢 Pros: Unbiased if true and easy to analyze
        - 🔴 Cons: Impractical due to requiring full list of population
    2. Stratified Sampling
        - Divide population in to homogenous subgroups (strata) and take random samples from each
          e.g: Divide by age
        - 🟢 Pros: Representation of all groups
        - 🔴 Cons: Requires complex design and knowledge to divide
    3. Systematic Sampling
        - Select every k-th item
        - 🟢 Pros: Easy, good coverage
        - 🔴 Cons: Hidden numerical patterns cause bias, also requires full list
    4. Cluster Sampling
        - Divide population into clusters and select one or more entire clusters
        - 🟢 Pros: Cheaper and convenient
        - 🔴 Cons: Less precise than stratified, bias due to only some clusters
    5. Multi Stage Sampling
        - Combines Several Sampling Methods
2. Non Probability Sampling
    - Individuals have either unknown or unequal chance of being selection

    1. Convenience Sampling
        - Sample whoever you can
        - 🟢 Pros: Cheap and convenient
        - 🔴 Cons: Extremely Biased
    2. Purposive/Judgmental Sampling`❌[Incomplete]`
    3. Quota Sampling`❌[Incomplete]`
    4. Snowball Sampling`❌[Incomplete]`
    5. Sampling Distributions
3. Over And Under Sampling: Handling imbalance in dataset by either choosing from lower representation class more or
   less from over-represented class to create balance.  
   Techniques:
    1. ### SMOTE: (Synthetic Minority Oversampling Technique)
        - Choose lower representation class. Plot similar to k means and take a subset of points and use their average
          value as new entry.
        - 🔴Not real data so might overfit
    2. ### ADASYN: (Adaptive Synthetic Sampling)
        - Handles imbalance in datasets but focuses on creating harder to classify samples
        - How?
            - Plot all points on graph, for each point in minority class find which minority points have max majority
              neighbours (mimicking KNN)
            - Using these decision boundary points perform SMOTE
        - 🟢 Teach the model more since focuses on creating points in difficult regions
        - 🔴 Can introduce noise

## Text Preprocessing

1. ### Lowercasing
    - Standardize text to lowercase
2. ### Tokenization
    - Splits text into smaller units (called tokens), usually words.
    - Types of tokenization:
        - Word tokenization: Splits text into individual words.
        - Character tokenization: Splits text into characters.
            - 🟢 Small vocab
            - 🔴 Too much to compute
            - 🔴 Doesn't capture context
        - Sentence tokenization: Splits text into sentences. (very low usecase)
        - Subword tokenization: Breaks down into smaller units than words (useful in deep learning).  
          Two Popular Types: (ℹ️ Covered in LLM.md)
            1. Byte Pair Encoding
            2. Word Piece Encoding
3. ### Stopword Removal
    - Removes common words that appear frequently but add little meaning (e.g., "the", "is", "and").
    - 🟢 Reduces dimensionality and focuses on meaningful words.
4. ### Stemming and Lemmatization
    - Stemming: Cuts words to their base form (may not be a real word).
    - Example: "running" → "run", "flies" → "fli"
    - Lemmatization: Converts words to their meaningful base form using vocabulary and grammar.
        - o Example: "running" → "run", "better" → "good"
    - Lemmatization is usually more accurate than stemming.
5. ### Others:
    - Removing Punctuation, Numbers, and Special Characters
    - Removing Duplicates, Blank Lines
    - Spelling Correction
    - Slang and Abbreviation Handling (Optional)
    - Expanding contractions (e.g., "can't" → "cannot")
    - Ensures consistency across the dataset.
6. ### Vectorization
    - Converts processed text into numerical format.
    - Common methods:
        1. Bag of Words (BoW): Counts word frequency in each document.
        2. TF-IDF (Term Frequency-Inverse Document Frequency): Weighs words by importance.
        3. Word Embeddings: Represents words in dense vector form based on meaning (e.g., Word2Vec, GloVe, BERT).

## Word Embedding

- Goal behind embeddings is to convert words into vectors and ensure two words/vectors of similar meaning/context are
  mapped close in the embedding space. usually calculated via cosine similarity or Euclidean distance
- Types:
    1. ## Static Embeddings [DEPRECATED]]

        - Created with a very large but finite corpus thus can only work with words in vocab.
        - 🔴 Don't work well with polysemy. (words with multiple meanings)
          Notable Implementations:

        1. Word2Vec
           Trained using a neural network.  
           Similar to how encoder decoder is. Here W<sub>in</sub> and W<sub>out</sub> exist.
            1. CBOW (Continous Bag of Words) -> Predict middle word using surrounding
               input words fed through and multiplied to Win to embed and averaged and multiplied with Wout.
               Finally, softmax to get output prob and calculate gradient against actual word
            2. SkipGram -> Predict Surrounding words using middle word
               ⚠️Figure out how one embedding turns into multiple
        2. Glove (Global Vectors for Word Representation)  
           Works using global occurrence of words
           Create a co-occurrence matrix i.e. words on both axis and occurrence probablity on each i,j
           Then $$w_i^T \tilde{w}_j + b_i + \tilde{b}_j \approx \log(X_{ij})$$
           i.e. the embedding of the two words + bias terms should match their co-occurrence.
           Deep network trained W using this as loss.  
           🟢 Produces embeddings where linear relationships capture meaning
           Example: king - man + woman = queen

    2.  ## Dynamic Embeddings

    - Dynamic embeddings are embeddings that can change depending on context.
    - <table><tr><th>Model</th><th>How it works</th> </tr>
      <tr><td>ELMo</td><td>Uses a deep LSTM to generate embeddings based on the entire sentence</td> </tr>
        <tr><td>BERT</td><td>Uses transformers and attention to create context-aware embeddings for each word</td> </tr>
        <tr><td>GPT</td><td>Also produces dynamic embeddings (during generation or fine-tuning)</td> </tr>
          </table>

1. Contextual -> Used in attention/Transformers

## **Word Embedding in Non-Textual Context**

- **Item2Vec** is used to generate dense vector representations of items (e.g., songs, movies, books)
- Wav2Vec: Converts audio into structures similar to how sentences are.




