
  

# Machine Learning Topics

## Data Processing

1.  **Data Preparation**

2.  **Data Imbalance Consequences**

3.  **SMOTE**

4.  **Feature Selection**

5.  **Sampling**

6.  **Normalization**

7.  **Text Preprocessing**

		1. Lowercasing
			 • Converts all text to lowercase.
			 • Helps standardize words (e.g., "Apple" and "apple" are treated the same).
		2. Tokenization
			• Splits text into smaller units (called tokens), usually words.
			• Types of tokenization:
				o Word tokenization: Splits text into individual words.
				o Sentence tokenization: Splits text into sentences.
				o Subword tokenization: Breaks down into smaller units than words (useful in deep learning).
		3. Removing Punctuation, Numbers, and Special Characters
			• Eliminates characters that may not be useful for analysis.
			• Keeps only alphabetical text to reduce noise.
		4. Stopword Removal
			• Removes common words that appear frequently but add little meaning (e.g., "the", "is", "and").
			• Reduces dimensionality and focuses on meaningful words.
		5. Stemming and Lemmatization
			• Both reduce words to their base or root form.
			• Stemming: Cuts words to their base form (may not be a real word).
				o Example: "running" → "run", "flies" → "fli"
			• Lemmatization: Converts words to their meaningful base form using vocabulary and grammar.
				o Example: "running" → "run", "better" → "good"
			• Lemmatization is usually more accurate than stemming.
		6. Text Normalization
			• Standardizes text in various ways, such as:
				o Expanding contractions (e.g., "can't" → "cannot")
				o Handling misspellings
				o Converting accented characters
			• Ensures consistency across the dataset.
		7. Removing Duplicates and Blank Lines
			• Eliminates repeated or empty entries.
			• Helps in maintaining a clean dataset.
		8. Spelling Correction (Optional)
	  		• Fixes typos and spelling errors to improve word consistency.
			• Especially useful for user-generated content (like social media posts).
		9. Slang and Abbreviation Handling (Optional)
			• Translates informal words to standard language.
				o Example: "u" → "you", "btw" → "by the way"
		10. N-gram Generation
			• Captures combinations of words (e.g., bigrams like "not good").
			• Helps preserve the context and order of words.
			• Useful when word combinations matter more than individual words.
		11. Final Step: Vectorization
			• Converts processed text into numerical format.
			• Common methods:
				o Bag of Words (BoW): Counts word frequency in each document.
				o TF-IDF (Term Frequency-Inverse Document Frequency): Weighs words by importance.
				o Word Embeddings: Represents words in dense vector form based on meaning (e.g., Word2Vec, GloVe, BERT).

8. **Padding and Truncation of Text**
Padding and truncation are important steps in text preprocessing when you're working with machine learning models that require fixed-length input, especially neural networks like RNNs, LSTMs, or transformers (like BERT). 
	- Padding
		- What it does: Adds extra tokens (usually zeros or a special padding token) to the end or beginning of a sequence to make it a fixed length. 
		- Why it's needed: Not all sentences are the same length, but models need inputs of the same size. 
		- Where it's used: Often used after tokenization and before feeding data into a model.
	- Truncation 
		- What it does: Cuts longer sequences down to the maximum allowed length. 
		- Why it's needed: To avoid memory issues and keep computation efficient. 
		- How it's done: Usually from the end (but can be from the start depending on the use case).

Common Settings (when using libraries like TensorFlow or Hugging Face):
 - padding='max_length': Pads all sequences to the same length.
 - truncation=True: Automatically truncates sequences longer than the max length.
 - max_length=128: Defines the maximum sequence length.
 
11. **Word Embedding**
	- **Goal**: The goal behind embeddings is to convert words into vectors and ensure two words/vectors of similar meaning/context are mapped close in the embedding space.
	i.e Calculating the dot product (i.e angle between the two should be small) should be large. 
	- **Word Embedding**: An embedding is a way to represent words as dense vectors in a lower-dimensional space, where similar words end up close together. 
Think of each dimension in the embedding as capturing some latent feature — like "animal-ness", "femininity", "past tense-ness", etc.
These features are not predefined by us — the model learns them during training. It is like asking the model to classify words along N different (hidden) axes, where N is the embedding size.
	1. One Hot Encoding 
	2. Distributed Representations 
	3. Static Embeddings [DEPRECATED] 
Static embeddings are the oldest type of word embedding. The embeddings are generated against a large corpus but the number of words, though large, is finite. If you have a word whose embedding needs to be looked up that was not in the original corpus, then you are out of luck. In addition, a word has the same embedding regardless of how it is used, so static embeddings cannot address the problem of polysemy, that is, words with multiple meanings.
	4. Dynamic Embedding Dynamic embeddings are word representations that change depending on the context they appear in. This is in contrast to static embeddings, where each word has a single fixed vector, no matter where or how it appears. 
	
		|   Model             |How it works | 
		|----------------|-------------------------------|-----------------------------| 
		|ELMo|Uses a deep LSTM to generate embeddings based on the entire sentence|
		|BERT|Uses transformers and attention to create context-aware embeddings for each word|
		|GPT|Also produces dynamic embeddings (during generation or fine-tuning)|
	5. **Word2Vec**: turns words into dense vectors of numbers, so that similar words have similar vectors. These vectors capture semantic meaning — for example: 
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
Captures meaning and relationships between words. There are two main architectures:
		 1.	 CBOW (Continuous Bag of Words): Predicts a word given its context (surrounding words). 
 Example: "The cat sat on the ___" → model predicts "mat".
 In Word2Vec (CBOW), the context window is the input to the model. 
 
		 2.	Skip-Gram: Predicts context words from a given word. 
Example: Input "cat" → model predicts words like "the", "sat", "on".
In Skipgram we just pass one word

	**Characteristics of Word2Vec Embeddings***: 
	1.	Dense: 
		- Unlike one-hot vectors (which are mostly 0s), Word2Vec embeddings are packed with real numbers.
		- Example: A one-hot vector for "cat" might be 10,000 dimensions long with only one "1". In contrast, a Word2Vec embedding might be 100-300 dimensions, all with meaningful values like [0.25, -1.3, 0.7, ..., 0.01]. 
	2.	Low-dimensional: 
		- Usually 50–300 dimensions, which is much smaller and more computationally efficient than sparse vectors.
	3.	Semantic meaning is encoded: 
		- Words that appear in similar contexts (e.g., “king” and “queen”) will have similar vectors. 
		- The relationships between vectors capture analogies (like we mentioned: king - man + woman ≈ queen).
	4.	Fixed size: 
		- Each word, regardless of how common or rare, gets a vector of the same length. 
	5.	Context-independent:
		- Each word has one vector, no matter the sentence. So "bank" (as in river bank vs money bank) has the same embedding in both contexts. 
		- This is a limitation that newer models like BERT try to solve. 
		
	6. **Glove** 
	GloVe differs from Word2Vec in that Word2Vec is a predictive model while GloVe is a count-based model. The first step is to construct a large matrix of (word, context) pairs that co-occur in the training corpus. Rows correspond to words and columns correspond to contexts, usually a sequence of one or more words. Each element of the matrix represents how often the word co-occurs in the context. The GloVe process factorizes this co-occurrence matrix into a pair of (word, feature) and (feature, context) matrices. The process is known as matrix factorization and is done using Stochastic Gradient Descent (SGD), an iterative numerical method. R = P * Q ≈ R’ 
	Thus, the model decomposes a larger matrix R into it’s constituents approximately. The difference between the matrices R and R’ represents the loss and is usually computed as the mean-squared error between the two matrices.
	 The GloVe process is much more resource-intensive than Word2Vec. This is because Word2Vec learns the embedding by training over batches of word vectors, while GloVe factorizes the entire co-occurrence matrix in one shot.
	 7. **Fastext** 
	 8. **Concept Number batch**

12. **Character Embedding**
13. **Sentence Embedding**
14. **Language Based Model**
15. **Word Embedding in Non-Textual Context**
16.  




________________________________________

14. 





________________________________________

15. 




________________________________________

16. 




________________________________________

17. 


