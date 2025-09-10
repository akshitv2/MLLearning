# Data Pre-Processing

## **Feature Selection**

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

Goal behind embeddings is to convert words into vectors and ensure two words/vectors of similar meaning/context are
mapped close in the embedding space. usually calculated via cosine similarity or Euclidean distance

## Static Embeddings [DEPRECATED]

Created with a very large but finite corpus thus can only work with words in vocab.  
🔴 Don't work well with polysemy. (words with multiple meanings)

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
   Create a co-occurrence matrix i.e words on both axis and occurrence probablity on each i,j
   Then $$w_i^T \tilde{w}_j + b_i + \tilde{b}_j \approx \log(X_{ij})$$
    i.e the embedding of the two words + bias terms should match their co-occurrence.
   Deep network trained W using this as loss.  
   🟢 Produces embeddings where linear relationships capture meaning
   Example: king - man + woman = queen

## Dynamic Embeddings
Dynamic embeddings are embeddings that can change depending on context.
1. Contextual -> Used in attention/Transformers

## **Word Embedding in Non-Textual Context**
- **Item2Vec** is used to generate dense vector representations of items (e.g., songs, movies, books)
- Wav2Vec: Converts audio into structures similar to how sentences are.




