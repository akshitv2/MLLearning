# LLM
## Index

- [LLM](#llm)
  - [Foundations and basics](#foundations-and-basics)
    - [n-grams](#n-grams)
    - [Bag of Words](#bag-of-words)
    - [TF-IDF](#tf-idf)
    - [Modern Tokenization](#modern-tokenization)
      - [Byte Pair Encoding](#byte-pair-encoding)
      - [Word Piece Encoding](#word-piece-encoding)
    - [Positional Encoding](#positional-encoding)
      - [Usage](#usage)
      - [Sinusoidal](#sinusoidal)
      - [Learned](#learned)
    - [Contrastive Learning](#contrastive-learning)
      - [Contrastive Learning in LLMS](#contrastive-learning-in-llms)
  - [Large Language Models](#large-language-models)
    - [Training Process](#training-process)
      - [Pre-Training](#pre-training)
      - [Fine-Tuning](#fine-tuning)
    - [Other Types](#other-types-of-llms)
      - [Multimodal Large Language Models (MLLMs)](#multimodal-large-language-models-mllms)
      - [Agentic Systems](#agentic-systems)
      - [Advanced Reasoning Models](#advanced-reasoning-models)
        - [Key Techniques](#key-techniques)
          - [Chain of thought Prompting](#chain-of-thought-prompting)
          - [Self Consistency](#self-consistency)
          - [Tool Use](#tool-use)
          - [Tree/Graph Based Reasoning](#treegraph-based-reasoning)
          - [Using RLHF (Reinforcement learning from human feedback)/ RLAIF (Reinforcement learning from ai feedback)](#using-rlhf-reinforcement-learning-from-human-feedback-rlaif-reinforcement-learning-from-ai-feedback)
      - [Small Language Models (SMLs)](#small-language-models-smls)
  - [RAG : Retrieval Augmented Generation](#rag--retrieval-augmented-generation)
    - [Composed of](#composed-of)
      - [Retriever](#retriever)
      - [Generator](#generator)
    - [Working Process](#working-process)
      - [Indexing Phase (The preparation)](#indexing-phase-the-preparation)
      - [Retrieval and Generation](#retrieval-and-generation)
    - [Search Algorithms](#search-algorithms)
      - [Sparse Retrieval](#sparse-retrieval)
        - [Common Examples](#common-examples)
          - [BM25](#bm25)
      - [Dense Retrieval](#dense-retrieval)
        - [Dual Encoders/Bi-encoders](#dual-encodersbi-encoders)
        - [Similarity Search](#similarity-search)
        - [Notable Implementations](#notable-implementations)
          - [Dense Passage Retrieval (DPR)](#dense-passage-retrieval-dpr)
          - [ColBert](#colbert)
          - [Contriever](#contriever)
          - [RocketQA](#rocketqa)
        - [Algorithms Used](#algorithms-used)
          - [Hierarchical Navigable Small World (HNSW)](#hierarchical-navigable-small-world-hnsw)
          - [FAISS (Facebook AI Similarity Search)](#faiss-facebook-ai-similarity-search)
          - [ScaNN (Scalable Nearest Neighbors)](#scann-scalable-nearest-neighbors)
      - [Hybrid Search](#hybrid-search)
      - [Recursive Retrieval](#recursive-retrieval)
  - [Prompt Engineering](#prompt-engineering)
    - [Techniques](#techniques)
      - [Zero-Shot Prompting](#zero-shot-prompting)
      - [Few Shot Prompting](#few-shot-prompting)
      - [Chain-of-Thought (CoT) Prompting](#chain-of-thought-cot-prompting)
      - [Role-Playing](#role-playing)
      - [Instruction Tuning](#instruction-tuning)
    - [Challenges](#challenges)
      - [Catastrophic Forgetting](#catastrophic-forgetting)
        - [Catastrophic Forgetting Solutions](#catastrophic-forgetting-solutions)
          - [Replay](#replay)
          - [Elastic Weight Consolidation](#elastic-weight-consolidation)
          - [Dynamic Architectures](#dynamic-architectures)
      - [Hallucination](#hallucination)
        - [Hallucination Solution](#hallucination-solution)
          - [Retrieval Augmented Generation](#retrieval-augmented-generation)
          - [Training with uncertainty in answer](#training-with-uncertainty-in-answer)
          - [Use LLMs that can use symbolic reasoning](#use-llms-that-can-use-symbolic-reasoning)
          - [Chain of thought](#chain-of-thought)
          - [let AI use tools like calculator](#let-ai-use-tools-like-calculator)
      - [Repetition](#repetition)
      - [Degeneration](#degeneration)
      - [Adversarial Prompts](#adversarial-prompts)
        - [Solutions](#solutions)
          - [Instruction Separation](#instruction-separation)
          - [Input and token validation](#input-and-token-validation)
          - [Classifier Layer](#classifier-layer)
          - [Safeguard Systems](#safeguard-systems)
          - [Human in Loop](#human-in-loop)
      - [Context Rot](#context-rot)
        - [Solution](#solution)
    - [Evaluation](#evaluation)
      - [Perplexity](#perplexity)
      - [BLEU: Bilingual Evaluation Understudy](#bleu-bilingual-evaluation-understudy)
        - [BP: Brevity Penalty](#bp-brevity-penalty)
        - [Score](#score)
      - [ROUGE: Recall Oriented Understudy for Gisting Evaluation](#rouge-recall-oriented-understudy-for-gisting-evaluation)
        - [ROUGE-N](#rouge-n)
        - [ROUGE-L](#rouge-l)
        - [ROUGE-W](#rouge-w)
        - [ROUGE-S/SU](#rouge-ssu)
      - [Recall, Precision, F1Score](#recall-precision-f1score)
      - [METEOR: Metric for Evaluation of Translation with Explicit Ordering](#meteor-metric-for-evaluation-of-translation-with-explicit-ordering)
    - [Evaluation Specific to Tasks](#evaluation-specific-to-tasks)
      - [Code](#code)
        - [Exact Match](#exact-match)
        - [Pass@k](#passk)
      - [Truthfulness](#truthfulness)
        - [TruthfulQA](#truthfulqa)
      - [General: MMLU (Massive Multitask Language Understanding)](#general-mmlu-massive-multitask-language-understanding)
      - [Maths and reasoning: GSM8K, MATH](#maths-and-reasoning-gsm8k-math)
      - [Fact Check: FEVER, FAST-CC, SciFact](#fact-check-fever-fast-cc-scifact)
    - [RAG Metrics](#rag-metrics)
      - [Recall@k](#recallk)
      - [Precision@K](#precisionk)
      - [NDCG: (Normalized Discounted Cumulative Gain)](#ndcg-normalized-discounted-cumulative-gain)
      - [MRR: Mean Reciprocal Rank](#mrr-mean-reciprocal-rank)
    - [Parameter Efficient Fine-Tuning](#parameter-efficient-fine-tuning)
      - [Full Fine-Tuning](#full-fine-tuning)
      - [Adapters](#adapters)
      - [Prompt Tuning](#prompt-tuning)
      - [LoRA (Low Rank Adaption)](#lora-low-rank-adaption)
      - [Prefix Tuning](#prefix-tuning)
    - [RAG Specific](#rag-specific)
      - [Vector DBS](#vector-dbs)
    - [Text Generation Strategies](#text-generation-strategies)
      - [Greedy decoding](#greedy-decoding)
      - [Beam Search](#beam-search)
      - [Top K Sampling](#top-k-sampling)
      - [Top P Sampling](#top-p-sampling)
      - [Temperature Scaling](#temperature-scaling)
    - [Human Centric Eval](#human-centric-eval)

# Foundations and basics

- #### n-grams
  n → No. of words  
  example: Hi I am Sam  
  then 1 gram: ["Hi","I","am","Sam"]  
  then 2 gram: [["Hi","I"],["I","am"],["am","Sam"]]  
  then 3 gram: [["Hi","I","am"],["I","am","Sam"]]  
  then 4 gram: ["Hi","I","am","Sam"]

- #### Bag of Words
  Let our lowercased vocab be ["the","cat","dog","sat","on","mat"]   
  then BOW converts sentence "The cat sat on the mat" to [2,1,0,1,1,1] (i.e. frequency of each word on the index in
  vocab).
  🔴Doesn't care about the word order
- #### **TF-IDF**
    - Term Frequency Inverse Document Frequency
    - Tf(T,d) = count of times t appears in doc d/total number of terms in d
    - IDF(T,d) = log(Number of docs in corpus/Number of docs containing t)
    - TFIDF = TFxIDF
    - **Idea:** TF gives how much word appears but is biased to common words. IDF gives lower scores for common words
      i.e. Rare
      word in docs means probably significant presence.
- #### Modern Tokenization:
    - ##### Byte Pair Encoding:
        - Iteratively merge two most frequent pairs of symbols (starts from characters merging to
          form words).
        - Choose two most frequent pairs of characters that appear together until you reach required vocab so creates
          combinations of common words: eg Un happy  
          h e l l o → he l l o → hel l o → hell o → hello
    - ##### **Word Piece Encoding**:
        - Combines word based on maximizing log likelihood of sentences. Likelihood of characters
          alone grows at rate of (1/26)^c.
        - So a vocab forming increases probablity of words.  
          $$s^\ast(w) = \arg\max_{s \in \mathcal{S}_V(w)} \prod_{i=1}^{K(s)} P(u_i)
          = \arg\max_{s \in \mathcal{S}_V(w)} \sum_{i=1}^{K(s)} \log P(u_i)$$
- #### **Positional Encoding**
    - ##### Usage:
        - Added to deep/shallow embeddings to create encoded and embedded token
    - ##### Sinusoidal:
        - Encodes position in sentence as well dimensionally.
        - $${PE}_{p,2i} = \sin\!\left(\frac{p}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$
        - $${PE}_{p,2i+1} = \cos\!\left(\frac{p}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$
        - where d is the dimension of embedding and p position in sentence
            - bound to [-1,1] due to sin and cos functions
    - #### Learned:
        - Train a set of vectors to output embedding number
        - Since learned embeddings are not bound by sin and cos can cause embedding collision
    - ℹ️Note: Since each embedding is different for each position the same word at position 1 and position 2 will appear
      as different words to the model (but since sin and cos are [-1,1] not that different)
- #### **Contrastive Learning**
  Type of self supervised/supervised learning, teaches models to recognize similar and diff things. Learns by comparing
  examples.
  Sometime use labels, but can learn without them as well by instead augmenting same sample enough times so that model
  learns to generalize e.g. model learns two
  dogs are similar by augmenting each dog enough times.
    - #### Contrastive Learning in LLMS
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
            - RAG: Use vector embeddings to find the closest docs.  
              Similar docs → Close  
              Diff Docs → Far
            - Instruction Fine-Tuning: Model learns what is the output for instruction  
              i.e. the answer to "Translate to german: Hello" and "Hallo" are closer.

# Large Language Models

- Use Transformer Architecture, attention mechanisms
- ### Training Process:
    - ### Pre-Training:
        - LLMs fed massive text corpora to learn patterns, grammar, facts
    - ### Fine-Tuning:
        - Models further trained on specific tasks to improve targeted use case performance
- ### Other types of LLMs:
    - ## Multimodal Large Language Models (MLLMs)
        - Integrate vision, audio and even 3d data (basically whatever can be encoded into a common embedding space)
    - ## Agentic Systems
        - New paradigm where LLM agents don't just generate text but have agency -> Use tools, planning and reasoning
        - Frameworks: ReAct (Reasoning + Act)
    - ## Advanced Reasoning Models:
        - Optimized for multistep problem-solving rather than just fluent text. Extend base models
        - ## Key Techniques:
            1. #### Chain of thought Prompting
                - Model is guided to generate intermediate steps instead of final answer
                - for e.g. write out logic and calculations for a math problem
            2. #### Self Consistency
                - Model generates multiple answers instead of one and answers are aggregated (like voting/bagging)
                - Increases reliability
            3. #### Tool Use:
                - Calls external systems/tools like apis, calculators
            4. #### Tree/Graph Based Reasoning
                - Use tree/graph like structure instead of one linear chain of reasoning
                - Branches need to be evaluated
                - Most consistent one chosen
            5. #### Using RLHF (Reinforcement learning from human feedback)/ RLAIF (Reinforcement learning from ai feedback)
                - Reasoning validated by this feedback
    - ## Small Language Models (SMLs)
        - AI language models with fewer parameters and a more focused scope than Large Language Models (LLMs), making
          them efficient, cost-effective, and ideal for specific tasks and resource-constrained environments like
          smartphones and edge devices.

# RAG : Retrieval Augmented Generation

- RAG Combines information retrieval with LLMs to improve performance of AI models.
- Composed of:
    1. ## Retriever:
        - Searches large external knowledge bases like DB, internet, docs. to find relevant info
        - Retrieves top k most relevant documents
    2. ## Generator:
        - Produces answer using the queried documents
        - Simply an LLM you provide docs to

- ## Working Process
    1. Indexing Phase (The preparation)
        - Data Ingestion: Collect data
        - Chunking: Documents broken down to fit LLMs context window
        - Embedding: Convert into numerical representation
        - Vector DataBase: Embeddings stored in specialized db for vectors
    2. Retrieval and Generation
        - Query Embedding: User query is also embedded into same space
        - Vector Search: System performs similarity search in vector db to find most relevant document chunks
        - Prompt Augmentation: Retrieved chunks combined with original query and prompt is asked from LLM.
        - Response Generation: Regular working of LLM except it has supplemental info this time

- ### Search Algorithms:
    - #### Sparse Retrieval:
        - Common Examples:
            - BM25: Sparse retrieval → Lexical matching
                - Works using tf-idf with some normalization.
                - 🔴 Only matches exact tokens and struggles with synonyms
                - 🟢 Best performance at keyword heavy queries
                - 🟢 Very Fast
    - #### Dense Retrieval:
        - Uses Dense embeddings by converting both query and docs to same common embedding
        - 🟢 Understands semantic similarity and contextual meaning.
        - 🟢 Better performance in capturing meaning
        - 🔴Require expensive embedding model with more compute and memory req
        - 🔴Hard to interpret
        - Dual Encoders/Bi-encoders:
            - This is the most prevalent design.
            - It uses two separate, independent neural networks (often fine-tuned transformer models like BERT,
              Sentence-Transformers) to encode the query and the documents into the same vector space.
        - Similarity Search:
            - Queries vector DB to find document vectors using cosine similarity or dot product (usually)

        - Notable Implementations:
            - Dense Passage Retrieval (DPR): Uses two BERTs one, uses negatives to push non-relevant docs away
            - ColBert: Contextualized Late Interaction over BERT
                - Creates a vector for every token in the document as well as query and does similarity matching on
                  those two.
            - Contriever
            - RocketQA
        - Algorithms Used:
            - Dense ret algos are based on ANN (Approximate Nearest Neighbour) Search
            - **Hierarchical Navigable Small World (HNSW)**: Builds a multi layer graph to find nearest vector
            - **FAISS (Facebook AI Similarity Search)**: library of algorithms for efficient similarity
            - **ScaNN (Scalable Nearest Neighbors)**: Developed by Google, ScaNN is a library for ANN search that
              focuses on
              high-performance and is particularly effective for large-scale datasets.
    - #### Hybrid Search
        - Combining semantic search (using embeddings to find related meanings) with keyword search (like TF-IDF or
          BM25) to get the best of both worlds. This ensures that both exact matches and semantically similar
          information are retrieved.
    - #### Recursive Retrieval
        - More complex approach where the system retrieves information, uses it to generate a more detailed query, and
          then retrieves more information in an iterative process to build a comprehensive answer

# Prompt Engineering

Prompt engineering is the art and science of communicating effectively with an AI to get the desired output.

## Techniques:

1. ### Zero-Shot Prompting:
    - Basic form of prompt, giving AI direct instruction without any other info or examples
2. ### Few Shot Prompting:
    - Provide AI with a few examples of desired input output format before asking final question.
        - example:
            - ```vbnet
                Translate English to French:
                English: Hello
                French: Bonjour
                English: How are you?
                French: Comment ça va?
                English: Good night
                French:
3. ### Chain-of-Thought (CoT) Prompting:
   - Model is guided to generate intermediate steps instead of final answer
4. ### Role-Playing 
   - This involves assigning a specific persona or role to the AI. This helps to set the tone, style, and context of the conversation.
5. ### Instruction Tuning
   - Providing a list of detailed rules or instructions for the AI to follow.

## Challenges

- ### Catastrophic Forgetting**:
    - Model forgets previously learned information when trained on new task.
    - ### Catastrophic Forgetting Solutions:
        1. Replay: Store examples of previous task and retrain on them alongside new task
        2. Elastic Weight Consolidation: (Regularization):
            - Uses Fisher Information Matrix.
            - if fim score ↑ then importance ↑ so penalizes more.
            - ℹ️ Note: FIM: Basically squared probablity of getting ∂(y<sub>actual</sub> | x)/∂(this nueron)
        3. Dynamic Architectures: Add neurons or layers for giving model space to learn new tasks.
- ### Hallucination
    - Model generates incorrect outputs with fluency and confidence. Since LLM are prediction models if they don't know
      a fact they predict the next best fitting words.
    - ### Hallucination Solution:
        1. Retrieval Augmented Generation: Use context to answer
            - While training, you can another LLM as a judge to evaluate correctly deriving info from RAG
        2. Training with uncertainty in answer i.e. LLM is allowed to say it doesn't know (combine with RLHF)
        3. Use LLMs that can use symbolic reasoning  
           ℹ️Note: Symbolic Reasoning: Manipulating symbols, let LLM come to an answer than know it. Ways to do it?
        4. Chain of thought
        5. let AI use tools like calculator

- ### **Repetition:**:
    - The model repeats the same words because probablity distribution peaks too sharply there.
- ### **Degeneration
    - Model gives bad outputs in general for e.g. repetitions, too many I don't know or yes, interesting. Phrases that
      don't provide quality signal.
    - Solution? Increase Temperature if repetition is occurring due to too stringent prediction limit, Repetition
      Penalties, RLHF is always an option
- ### Adversarial Prompts:
    - Models can be tricked by certain malicious prompts.
    - Solutions?
        - **Instruction Separation:**
            - Keep the system prompt (instructions the model must follow) separate and immutable in your application
            - Give system prompts much higher priority to prevent overriding
        - **Input and token validation:**
            - Tokenization: Keep only limited range of chars
            - Validation: Clean inputs of adversarial instructions like "ignore previous instructions"
        - **Classifier Layer**: Run a fast binary classifier to detect likely prompt-injection content before passing it
          to the LLM.
        - **Safeguard Systems**: Build safeguards against info stored in RAG and access to tools which check system
          identity as well
        - **Human in Loop**: Add Human approval for high risk queries
- ### Context Rot:
    - Degeneration as context window gets larger.
    - LLMs especially earlier ones were often trained on small sequence sets.
    - Solution:
        - Summarize text as it goes out of immediate window (can use deep embeddings or text summary)

## Evaluation

1. #### Perplexity
    - Measures how much model was perplexed by input (lower is better).
    - Actually how likely was this prediction.
    - $$\text{Perplexity} = \exp\Bigg(- \frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_1, w_2, \dots, w_{i-1}) \Bigg)$$
    - Notice how it's just average log likelihood, and it's just probablity of getting the word i when we have predicted
      till i-1. (So taken at every word)
2. #### BLEU: Bilingual Evaluation Understudy
    - Calling output sequence candidate here
    - ℹ️Bilingual: Because it was used for comparing translations from machine and human
    - Calculated in two parts:
        1. #### BP: Brevity Penalty
            - if candidate sequence length is shorter than reference penalizes it
            - $$\mathrm{BP} = \begin{cases}1, & \text{if } c > r,\\[6pt]\exp\!\left(1 - \dfrac{r}{c}\right), & \text{if } c \le r.\end{cases}$$
        2. #### Score:
            - Calculated as overlap of n grams (maxed out at ref text ngram frequency to discourage repitition more than
              needed).
            - p<sub>n</sub> = no. of times n gram appears in candidate (maxed at appearance in ref)/no. of times n gram
              appears
              in candidate.
            - Together:
            - $$\text{BLEU} = BP \cdot \exp\Bigg( \sum_{n=1}^{N} w_n \log p_n \Bigg)$$
            - Here we are taking geometric mean (w can be equal but also higher for higher ns). Essentially log sum
              becomes
              product and ln cancels out exp/

        - 🔴 Bad at synonyms/semantically same sentences
3. #### **ROUGE**: Recall Oriented Understudy for Gisting Evaluation
    - How much of reference is captured by candidate (or how well can candidate recall)
    - ROUGE-N: use ngrams too
    - = no. of grams that appear in candidate(maxed out at n grams in
      ref)/no. of grams that appear in **reference**
    - ROUGE-L: Measures longest common subsequence
        - Reference: "The cat sat on the mat"
        - Candidate: "Cat sat on mat"
        - LCS = "cat sat on mat" → length = 4
    - ROUGE-W: (Weighted LCS) Divide by Length of C or R
    - ROUGE-S/SU: Allows skips
4. #### **Recall, Precision, F1Score**: Use N grams
    - Precision: No. of correct gen ngrams/total n grams gen
    - Recall: No. of correct gen ngrams/total n grams in ref
    - F1 Score: 2 * (Precision * Recall)/(Precision + Recall)
5. #### **METEOR**: Metric for Evaluation of Translation with Explicit Ordering
    - Works by having a more lenient matching, as it matches:
        1. Synonyms
        2. Stemmed words eg. running run
        3. Sometimes semantic match
    - Then Calculates Precision Recall then F1 except usually weights Recall more.

## Evaluation Specific to Tasks

1. #### Code:
    1. #### Exact Match:
        - No differences allowed except variable names.
        - ⚠️Overly strict since code can have differences for same logic
    2. #### Pass@k:
        - Measures probablity least some of the top k probabilistic outputs is correct
        - Rewards model if it's mostly correct instead of a harsh penalty.
        - pass@k = 1 - no. of ways of choosing incorrect outputs/no. of ways of choosing outputs
        - i.e 1 - <sup>n-c</sup>C<sub>k</sub>/<sup>n</sup>C<sub>k</sub>
        - Example: Model outputs (5 samples):
            - def add(x,y): return x+y ✅
            - def add(a,b): return a+b ✅
            - def add(a,b): return a-b ❌
            - def add(a,b): return b+a ✅
            - def add(a,b): return a*b ❌
            - Pass@1k = 1 - 2c1/5c1 = 0.6
            - Pass@3k = 1 - 2c3/5c3 = 1 - 0 = 1
2. #### Truthfulness:
    - TruthfulQA: Benchmark dataset for evaluating truthfulness of LLMs
        - Covers cases which tempt models into giving false answers i.e. cases where humans are often misinformed
3. General: MMLU (Massive Multitask Language Understanding)
4. Maths and reasoning: GSM8K, MATH
5. Fact Check: FEVER, FAST-CC, SciFact

## RAG Metrics

- For recall and precision R = Relevant Docs, S<sub>k</sub>= Top k retrieved docs
- #### Recall@k:

$$\text{Recall@k} = \frac{|R \cap S_k|}{|R|}$$

- #### Precision@K:

$$\text{Precision@k} = \frac{|R \cap S_k|}{k}$$

- #### NDCG: (Normalized Discounted Cumulative Gain)
  Search retrieval ranking algo.  
  Basically you assign a relevance score to each doc (higher = more relevant) and divide by log(i+1) where i is how
  retrieval ranks it 1-inf (since small i = small denom. i.e higher score)  
  You sum up these scores. If ideal ordered followed then score would be high.   
  So you compare NDCG<sub>Predicted</sub>/NDCG<sub>ideal</sub>
  $$\text{NDCG@k} = \frac{\sum_{i=1}^{k} \frac{2^{rel_i}-1}{\log_2(i+1)}}{\sum_{i=1}^{k} \frac{2^{rel_i^\text{ideal}}-1}{\log_2(i+1)}}$$
- #### MRR: Mean Reciprocal Rank
  Ranks how quickly relevant item appears in a list (done over Q times)  
  Reciprocal Rank is 1/rank of first relevant doc
  $$\text{MRR} = \frac{1}{Q} \sum_{i=1}^{Q} \frac{1}{\text{rank}_i}$$

  ## Parameter Efficient Fine-Tuning
    - ### Full Fine-Tuning:E
        - 🔴 Expensive to compute
        - 🔴 High risk of catastrophic forgetting especially when pretrained model Dataset not available to you to
          revisit
    - ### Adapters:
        - Small trainable set of layers introduced in a frozen model
    - ### Prompt Tuning:
        - Small set of soft prompts i.e. not a full layer in the arch but instead just embedding added to the
          start of user's prompt (usually torch.nn.Embedding). Model is frozen throughout
    - ### LoRA (Low Rank Adaption) :
        - Reduces no. of trainable parameters by learning low rank A and B matrix instead of same
          rank as the complex model. Lora modifies the W matrices which sit before attention calc  
          Q=XWQ​,K=XWK​,V=XWV  
          For e.g: ΔW=AB, and W<sub>final</sub>​= W + ΔW  
          so W is not modified at all
    - ### Prefix Tuning: to be added

  ## RAG Specific
    - ### Vector DBS
    -

## Text Generation Strategies

- **Greedy decoding**: Choose simply highest probablity
- **Beam Search**: Instead of choosing directly most probable outputs you explore top k probablities at every step
  essentially becoming a tree.  
  Can choose between k generated trees:
    - Maximum log likelihood (of entire sequence) since later tokens might push up probablity  
      🟢 Simple to implement  
      🟢 More optimal than greedy decoding
      🔴 Doesn't guarantee optimal solution
- Top K Sampling: Instead of picking top probablity sample from top K (actually ignores >k and normalizes and picks
  randomly based on probablity k)
    - Higher K = more randomness
- Top P Sampling: Instead of picking top probablity pick shortest combinations >=p. Actually at each step creates a pool
  of candidates which combine >=p and choose one of them.
  Why? Because top k fails if top 5 only cover 20% i.e a lot of reasonable options
- Temperature Scaling: Controls sharpness of probablity distribution before sampling. Higher temp = Flatter Dist
  $$P_i^{(\tau)} = \frac{\exp\!\left(\frac{z_i}{\tau}\right)}{\sum_j \exp\!\left(\frac{z_j}{\tau}\right)}$$
-
    - Flatter dist = higher chance for less probablity to be picked
    - Sharper dist = higher chance for high probability to be picked (greedy)

## Human Centric Eval








