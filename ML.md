
# Machine Learning Topics 
## Data Processing
**1.	Data Preparation
2.	Data Imbalance Consequences
3.	SMOTE
4.	Feature Selection
5.	Sampling
6.	Normalization
7.	Text Preprocessing**
	1. Lowercasing
		•	Converts all text to lowercase.
		•	Helps standardize words (e.g., "Apple" and "apple" are treated the same).
________________________________________
**8. Tokenization**
•	Splits text into smaller units (called tokens), usually words.
•	Types of tokenization:
o	Word tokenization: Splits text into individual words.
o	Sentence tokenization: Splits text into sentences.
o	Subword tokenization: Breaks down into smaller units than words (useful in deep learning).
________________________________________
9. Removing Punctuation, Numbers, and Special Characters
•	Eliminates characters that may not be useful for analysis.
•	Keeps only alphabetical text to reduce noise.
________________________________________
10. Stopword Removal
•	Removes common words that appear frequently but add little meaning (e.g., "the", "is", "and").
•	Reduces dimensionality and focuses on meaningful words.
________________________________________
11. Stemming and Lemmatization
•	Both reduce words to their base or root form.
•	Stemming: Cuts words to their base form (may not be a real word).
o	Example: "running" → "run", "flies" → "fli"
•	Lemmatization: Converts words to their meaningful base form using vocabulary and grammar.
o	Example: "running" → "run", "better" → "good"
•	Lemmatization is usually more accurate than stemming.
________________________________________
12. Text Normalization
•	Standardizes text in various ways, such as:
o	Expanding contractions (e.g., "can't" → "cannot")
o	Handling misspellings
o	Converting accented characters
•	Ensures consistency across the dataset.
________________________________________
13. Removing Duplicates and Blank Lines
•	Eliminates repeated or empty entries.
•	Helps in maintaining a clean dataset.
________________________________________
14. Spelling Correction (Optional)
•	Fixes typos and spelling errors to improve word consistency.
•	Especially useful for user-generated content (like social media posts).
________________________________________
15. Slang and Abbreviation Handling (Optional)
•	Translates informal words to standard language.
o	Example: "u" → "you", "btw" → "by the way"
________________________________________
16. N-gram Generation
•	Captures combinations of words (e.g., bigrams like "not good").
•	Helps preserve the context and order of words.
•	Useful when word combinations matter more than individual words.
________________________________________
17. Final Step: Vectorization
•	Converts processed text into numerical format.
•	Common methods:
o	Bag of Words (BoW): Counts word frequency in each document.
o	TF-IDF (Term Frequency-Inverse Document Frequency): Weighs words by importance.
o	Word Embeddings: Represents words in dense vector form based on meaning (e.g., Word2Vec, GloVe, BERT).
