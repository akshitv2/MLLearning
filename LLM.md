# LLM

# Index

## Foundations
- [n-grams](#n-grams)
- [Bag of Words](#bag-of-words)
- [TF-IDF](#tf-idf)
- [Positional Encoding](#positional-encoding)
- [Contrastive Learning](#contrastive-learning)
  - [Contrastive Learning in LLMs](#contrastive-learning-in-llms)
    - [Types](#types)
      - Positive pairs
      - Semantic pairs
      - Instruction Response Pairs
    - [Applications](#applications)
      - Contrastive Pretraining for Embeddings
      - RAG
      - Instruction Fine Tuning

## Challenges
- [Catastrophic Forgetting](#catastrophic-forgetting)
  - Solutions
    1. Replay
    2. Elastic Weight Consolidation
    3. Dynamic Architectures
- [Hallucination](#hallucination)
  - Solutions
    1. Retrieval Augmented Generation
    2. Training with uncertainty
    3. Use symbolic reasoning
- [Repetition](#repetition)
- [Degeneration](#degeneration)
- [Adversarial Prompts](#adversarial-prompts)
- [Context Rot](#context-rot)

## Evaluation
- [Perplexity](#perplexity)
- [BLEU](#bleu)
- [ROUGE](#rouge)
- [Recall, Precision, F1Score](#recall-precision-f1score)
- [METEOR](#meteor)

### Evaluation Specific to Tasks
- [Code](#code)
  - Exact Match
  - Pass@k
- [Truthfulness](#truthfulness)
- [General](#general)
- [Maths and reasoning](#maths-and-reasoning)
- [Fact Check](#fact-check)

### RAG Metric


### Foundations
- #### n-grams
    n -> No. of words  
    example: Hi I am Sam  
    then 1 gram: ["Hi","I","am","Sam"]  
    then 2 gram: [["Hi","I"],["I","am"],["am","Sam"]]  
    then 3 gram: [["Hi","I","am"],["I","am","Sam"]]  
    then 4 gram: ["Hi","I","am","Sam"]  
    
- #### Bag of Words
  Let our vocab be ["the","cat","dog","sat","on","mat"]   
  then BOW converts sentence "The cat sat on the mat" to [2,1,0,1,1,1] (i.e frequency of each word on the index in vocab).
  🔴Doesn't care about the word order
- #### **TF-IDF**
  Term Frequency Inverse Document Frequency  
  Tf(T,d) = count of times t appears in doc d/total number of terms in d  
  IDF(T,d) = log(Number of docs in corpus/Number of docs containing t)  
  TFIDF = TFxIDF
  Idea: TF gives how much word appears but is biased to common words. IDF gives lower scores for common words i.e Rare word in docs means probably significant presence.
- #### Modern Tokenization:
  - **Byte Pair Encoding**: Iteratively merge two most frequent pairs of characters.  
  Choose two most frequent pairs of characters that appear together until you reach required vocab so creates combinations of common words: eg Un happpy  
  h e l l o -> he l l o -> hel l o -> hell o -> hello
  - **Word Piece Encoding**: Combines word based on maximizing log likelihood of sentences. Likelihood of characters alone grows at rate of (1/26)^c.
  So a vocab forming increases probablity of words.  
  $$s^\ast(w) = \arg\max_{s \in \mathcal{S}_V(w)} \prod_{i=1}^{K(s)} P(u_i)
= \arg\max_{s \in \mathcal{S}_V(w)} \sum_{i=1}^{K(s)} \log P(u_i)$$
- #### **Positional Encoding**
    Encodes position in sentence as well dimensionally.
  $${PE}_{p,2i}   = \sin\!\left(\frac{p}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$  
  $${PE}_{p,2i+1} = \cos\!\left(\frac{p}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$  
  where d is the dimension of embedding and p position in sentence
- #### **Contrastive Learning**  
    Type of self supervised learning, teaches models to recognize similar and diff things. Learns by comparing examples.
    Doesn't use labels, instead augments same sample enough times so that model learns to generalize e.g model learns two dogs are similar by augmenting each dog enough times.
    - **Contrastive Learning in LLMS**  
        - Types:  
        Inputs are text sequences, to create similarity pairs:
            - Positive pairs: (created using augmenting text)
                - replace words with synonyms
                - mask words
                - paraphrase
            - Semantic Pairs:
                - both sentences have same meaning (usually human labelled)  
                ⚠️ note: here it isn't self supervised
            - Instruction Response Pairs:
                - form pairs using question and their human answer
        - Applications:
            - Contrastive Pretraining for Embeddings: Used in OpenAi's ada & CLIP to produce dense embeddings.
            - RAG: Use vector embeddings to find closest docs.  
                Similar docs -> Close  
                Diff Docs -> Far  
            - Instruction Fine Tuning: Model learns what is the output for instruction  
            i.e the answer to "Translate to german: Hello" and "Hallo" are closer.

## Challenges
- #### **Catastrophic Forgetting**: Model forgets previously learned information when trained on new task.
    - Solutions: 
        1. Replay: Store examples of previous task and retrain on them alongside new task
        2. Elastic Weight Consolidation: (Regularization): Uses Fisher Information Matrix.
        if fim if fim score ↑ then importance ↑ so penalizes more.
        ℹ️ Note: FIM: Basically squared probablity of getting ∂(y<sub>actual</sub> | x)/∂(this nueron)
        3. Dynamic Architectures: Add neurons or layers for giving model space to learn new tasks.
- #### **Hallucination**: Model generates incorrect outputs with fluency and confidence. Since LLM are prediction models if they don't know a fact they predict the next best fitting words.
Solution:
    1. Retrieval Augmented Generation: Use context to answer + use another LLM as a judge to verify
    2. Training with uncertainity in answer i.e LLM is allowed to say it doesn't know (combine with RLHF)
    3. Use LLMs that can use symbolic reasoning  
    ℹ️Note: Symbolic Reasoning: Manipulating symbols, let LLM come to an answer than know it. Ways to do it?  
    - Chain of thought
    - let AI use tools like calculator
- #### **Repetition:**: The model repeats the same words because probablity distribution peaks too sharply there.
- #### **Degeneration**: Model gives bad outputs in general for eg. repititions, too many i don't know or yes, interesting. Phrases that don't provide quality signal.  
Solution? Increate Temperature, Repitition Penalties, RLHF is always an option
- **Adversarial Prompts**: Models can be tricked by certain malicious prompts.
- **Context Rot**: Degeneration as context window gets larger. LLMs are often trained on small sequence sets.  
💡Solution? Summarize text as it goes out of immediate window (can deep embed or text summary)
            
## Evaluation
- #### **Perplexity**: Measures how much model was perplexed by input (lower is better). Actualy how likely was this prediction.
  $$\text{Perplexity} = \exp\Bigg(- \frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_1, w_2, \dots, w_{i-1}) \Bigg)$$
  Notice how it's just average log likelihood, and it's just probablity of getting the word i when we have predicted till i-1. (So taken at every word)
- #### **BLEU**: Billingual Evaluation Understudy  
    Output is called candidate here 
    ℹ️Billingual Because it was used for comparing translations from machine and human
    <br>Calculated in two parts:
    1. BP: Brevity Penalty<br> if reference corpus is larger than candidate corpus penalizes it.  
    BP = 1 if r<c else e<sup>(1-r/c)</sup> 
    2. Score: Calculated as overlap of n grams (maxed out at ref text ngram frequency to discourage repitition more than needed).
    p<sub>n</sub> = no. of times n gram appears in candidate (maxed at appearance in ref)/no. of times n gram appears in candidate.  
    Together:
    $$\text{BLEU} = BP \cdot \exp\Bigg( \sum_{n=1}^{N} w_n \log p_n \Bigg)$$
    Here we are taking geometric mean (w can be equal but also higher for higher ns). Essentially log sum becomes product and ln cancels out exp/
    - 🔴 Bad at synonyms/semantically same sentences
- #### **ROUGE**: Recall Oriented Understudy for Gisting Evaluation
    How much of reference is captured by candidate (or how well can candidate recall)
    - ROUGE-N: use ngrams too  
     = no. of grams that appear in candidate(maxed out at n grams in ref<br>--------------------------------------------<br>no. of grams that appear in **reference**
    - ROUGE-L: Measures longest common subsequence  
    Reference: "The cat sat on the mat"
    Candidate: "Cat sat on mat"  
    LCS = "cat sat on mat" → length = 4
    - ROUGE-W: (Weighted LCS) Divide by Length of C or R
    - ROUGE-S/SU: Allows skips
- #### **Recall, Precision, F1Score**: Use N grams
    - Precision: No. of correct gen ngrams/total n grams gen
    - Recall: No. of correct gen ngrams/total n grams in ref
    - F1 Score: 2 * (Prec * Recall)/(Prec + Recall)
- #### **METEOR**: Metric for Evaluation of Translation with Explicit Ordering  
Works by having a more lenient matching, as it matches:
1. Synonyms
2. Stemmed words eg. running run
3. Sometimes semantic match

Then Calculates Precision Recall then F1 except usually weights Recall more.

## Evaluation Specific to Tasks
- #### Code:
  - Exact Match: No differences allowed except variable names.
  ⚠️Overly strict since code can have differences for same logic
  - Pass@k: Measures probablity atleast some of the top k probablistic outputs is correct  
  i.e rewards model if it's mostly correct instead of a harsh penalty.
  pass@k = 1 - no. of ways of choosing incorrect outputs/no. of ways of choosing outputs  
  i.e 1 - <sup>n-c</sup>C<sub>k</sub>/<sup>n</sup>C<sub>k</sub>
  Example: Model outputs (5 samples):  
  def add(x,y): return x+y ✅  
  def add(a,b): return a+b ✅  
  def add(a,b): return a-b ❌  
  def add(a,b): return b+a ✅  
  def add(a,b): return a*b ❌   
  Pass@1k = 1 - 2c1/5c1 = 0.6
  Pass@3k = 1 - 2c3/5c3 = 1 - 0 = 1
- Truthfulness: TruthfulQA
- General: MMLU (Massive Multitask Language Understanding)
- Maths and reasoning: GSM8K, MATH
- Fact Check: FEVER, FAST-CC, SciFact
  
## RAG Metrics
- For recall and precision R = Relevant Docs, S<sub>k</sub>= Top k retrieved docs
- #### Recall@k:
 $$\text{Recall@k} = \frac{|R \cap S_k|}{|R|}$$
- #### Precision@K: 
$$\text{Precision@k} = \frac{|R \cap S_k|}{k}$$
- #### NDCG: (Normalized Discounted Cumulative Gain)  
    Search retrieval ranking algo.  
    Basically you assign a relevance score to each doc (higher = more relevant) and divide by log(i+1) where i is how retrieval ranks it 1-inf (since small i = small denom. i.e higher score)  
    You sum up these scores. If ideal ordered followed then score would be high.   
    So you compare NCDG<sub>Predicted</sub>/NCDG<sub>ideal</sub>
    $$\text{NDCG@k} = \frac{\sum_{i=1}^{k} \frac{2^{rel_i}-1}{\log_2(i+1)}}{\sum_{i=1}^{k} \frac{2^{rel_i^\text{ideal}}-1}{\log_2(i+1)}}$$
- #### MRR: Mean Reciprocal Rank  
  Ranks how quickly relevant item appears in a list (done over Q times)  
  Reciprocal Rank is 1/rank of first relevant doc
  $$\text{MRR} = \frac{1}{Q} \sum_{i=1}^{Q} \frac{1}{\text{rank}_i}$$

  ## Parameter Efficient Fine Tuning
  - Full Fine Tuning: Why NOT? Expensive to compute, high risk of catastrophic forgetting especially when pretrained model Dataset not available to you
  - Adapters: Small trainable set of layers intrdocued in a frozen model
  - Prompt Tuning: Small set of soft prompts i.e not a full layer in the arch but instead just embedding added to the start of user's prompt (usually torch.nn.Embedding). Model is frozen throughout
  - LoRA: Low Rank Adaption : Reduces no. of trainable parameters by learning low rank A and B matrix instead of same rank as the complex model.  Lora modifies the W matrices which sit before attention calc  
    Q=XWQ​,K=XWK​,V=XWV  
    For e.g: ΔW=AB, and W<sub>final</sub>​= W + ΔW  
    so W is not modified at all
  - Prefix Tuning:
  
  ## RAG Specific
   - Search Algorithms:
     - BM25: Sparse retrieval -> Lexical matching  
     Works using tf-idf with some normalization.  
     🔴Only matches exact tokens and struggles with synonyms  
     🟢Best performance at keyword heavy queries  
     🟢Very Fast  
    - ANN: Approximate Nearest Neighbour  
    Dense retrieval e.g FAISS, ScaNN  
    🔴Require expensive embedding model with more compute and memory req  
    🔴Hard to interpret  
     🟢Capture Semantic Similarity
     🟢 Better performance in capturing meaning  
    - ANN Implementations:
      - Use Cosine Similarity, Dot Product (If we don't want to normalize, so can capture magnitude not just direction), euclidean distance
      - HNSW: Builds a multilayer graph for efficient freedy search
  
  ## Text Generation Strategies
  - **Greedy decoding**: Choose simply highest probablity
  - **Beam Search**: Instead of choosing directly most probable outputs you explore top k probablities at every step essentially becoming a tree.  
  Can choose between k generated trees:  
    - Maximum log likelihood (of entire sequence) since later tokens might push up probablity  
    🟢 Simple to implement  
    🟢 More optimal than greedy decoding
    🔴 Doesn't guarantee optimal solution
  - Top K Sampling: Instead of picking top probablity sample from top K (actually ignores >k and normalizes and picks randomly based on probablity k)
    - Higher K = more randomness
  - Top P Sampling: Instead of picking top probablity pick shortest combinations >=p. Actually at each step creates a pool of candidates which combine >=p and choose one of them.
  Why? Because top k fails if top 5 only cover 20% i.e a lot of reasonable options 
  - Temperature Scaling: Controls sharpness of probablity distribution before sampling. Higher temp = Flatter Dist
  $$P_i^{(\tau)} = \frac{\exp\!\left(\frac{z_i}{\tau}\right)}{\sum_j \exp\!\left(\frac{z_j}{\tau}\right)}$$
  - 
    - Flatter dist = higher chance for less probablity to be picked
    - Sharper dist = higher chance for high problity to be picked (greedy)


  ## Human Centric Eval








