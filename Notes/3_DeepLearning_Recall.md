# 3 Deep Learning Recall

This list of questions is designed to test and strengthen your recall of key concepts from the document. Questions are grouped by major sections for easy navigation. Each question includes a link to the relevant subsection in the original document (using the anchors from the index). Answer them from memory first, then click the link to reread and verify.

## Basics

### Perceptron
1. What is the simplest type of artificial neural network, and how does it make predictions?[Reread](3_DeepLearning.md#perceptron)
   
2. List the four main components of a Perceptron (inputs, weights, bias, and the summation/activation steps).[Reread](3_DeepLearning.md#perceptron)
   
3. Write the formula for the summation function \( z \) in a Perceptron.[Reread](3_DeepLearning.md#perceptron)
   
4. Describe the step function used as the activation in a basic Perceptron, including its output cases.[Reread](3_DeepLearning.md#perceptron)

### Feed Forward Neural Networks
5. What are Feed Forward Neural Networks composed of, in terms of layers?[Reread](3_DeepLearning.md#feed-forward-neural-networks)

### Hidden Layers
6. What are Hidden Layers in a neural network?[Reread](3_DeepLearning.md#hidden-layers)

### Width of Model
7. What does the "Width of Model" refer to in neural networks?[Reread](3_DeepLearning.md#width-of-model)

### Weights and Biases
8. How are weights and biases defined in neural networks (including their roles)?[Reread](3_DeepLearning.md#weights-and-biases)

### Universal Approximation Theorem
9. State the Universal Approximation Theorem and what it implies (existence vs. efficiency).[Reread](3_DeepLearning.md#universal-approximation-theorem)

### Activation Function
10. Why are activation functions needed in neural networks?[Reread](3_DeepLearning.md#activation-function-1)
    
11. What is the Sigmoid activation function's formula, its bounds, and asymptotic behaviors? List its main pros and cons.[Reread](3_DeepLearning.md#sigmoid)
    
12. Compare Tanh to Sigmoid: formula, bounds, and key issues.[Reread](3_DeepLearning.md#tanh)
    
13. What is ReLU's formula, and what are its advantages and the "Dying ReLU" problem?[Reread](3_DeepLearning.md#relu-rectified-linear-unit)
    
14. How does Leaky ReLU address Dying ReLU, and what is its trade-off? Include the formula with \( \alpha \).[Reread](3_DeepLearning.md#leaky-relu)
    
15. What makes Parametric ReLU different from Leaky ReLU?[Reread](3_DeepLearning.md#parametric-relu)
    
16. Describe Swish activation: formula, behavior for large positive x, and its pros/cons.[Reread](3_DeepLearning.md#swish)
    
17. What is eLU's formula, and how does it improve on ReLU regarding smoothness and centering?[Reread](3_DeepLearning.md#elu-exponential-linear-unit)
    
18. When and why is Softmax used? Write its formula and explain why exponentials are used instead of raw values.[Reread](3_DeepLearning.md#softmax)
    
19. Why does the document answer the question: "Why is softmax \( e^{z_i} \) and not \( z_i \) when both sum to 1?"[Reread](3_DeepLearning.md#why-is-softmax-ez_i-and-not-z_i-when-both-sum-to-1)

### Gradient Descent
20. What is Gradient Descent, and write its update formula (including learning rate \( \eta \)).[Reread](3_DeepLearning.md#gradient-descent)
    
21. Define Epoch and Shuffling in the context of Gradient Descent.[Reread](3_DeepLearning.md#gradient-descent)
    
22. Compare Stochastic, Batch, and Minibatch Gradient Descent: pros, cons, and typical batch sizes.[Reread](3_DeepLearning.md#stochastic) | [Reread](3_DeepLearning.md#batch) | [Reread](3_DeepLearning.md#minibatch)
    
23. What is Momentum-Based Gradient Descent? Write the velocity update formulas with \( \gamma \) and \( \eta \).[Reread](3_DeepLearning.md#momentum-based)
    
24. How does Nesterov Accelerated Gradient Descent improve on standard Momentum? Include the key formula difference.[Reread](3_DeepLearning.md#nesterov-accelerated-gradient-descent)
    
25. What causes Vanishing Gradient in backpropagation, and list 4 solutions.[Reread](3_DeepLearning.md#vanishing-gradient)
    
26. Describe Exploding Gradient: cause and two common clipping methods for solutions.[Reread](3_DeepLearning.md#exploding-gradient)

### Backpropagation
27. What is Backpropagation? (Brief recall from context.)[Reread](3_DeepLearning.md#backpropagation)

### Weight Initialization
28. Why is Zero Initialization a terrible idea for weights?[Reread](3_DeepLearning.md#zero-init)
    
29. What problems does naive Random Initialization have?[Reread](3_DeepLearning.md#random-init-naive)
    
30. For Xavier Initialization, write the Uniform and Normal formulas, and for which activations is it ideal? Explain fan-in/fan-out.[Reread](3_DeepLearning.md#xavier-init)
    
31. Compare He Initialization to Xavier: formulas (Uniform/Normal), ideal activations, and why variance is doubled.[Reread](3_DeepLearning.md#he-init)

### Learning Rate Scheduling
32. Describe Step Decay: formula and behavior.[Reread](3_DeepLearning.md#step-decay)
    
33. What is Exponential Decay's formula and how does it differ from Step Decay?[Reread](3_DeepLearning.md#exponential-decay)
    
34. Explain Cosine Annealing: formula and why it's "gentler."[Reread](3_DeepLearning.md#cosine-annealing)
    
35. What is Polynomial Decay's formula?[Reread](3_DeepLearning.md#polynomial-decay)
    
36. How does "LR on Plateau" work?[Reread](3_DeepLearning.md#lr-on-plateau)
    
37. What are the pros/cons of Cyclical Learning Rates?[Reread](3_DeepLearning.md#cyclical)
    
38. Describe One Cycle scheduling and its benefit.[Reread](3_DeepLearning.md#one-cycle)

### Regularization
39. What is L1 (LASSO) regularization? Pros/cons and constraint shape.[Reread](3_DeepLearning.md#l1-lasso)
    
40. Explain L2 (Ridge) regularization: penalty type, why no sparsity, and constraint shape.[Reread](3_DeepLearning.md#l2-ridge)
    
41. What is Elastic Net, and its main drawback?[Reread](3_DeepLearning.md#elastic-net)
    
42. How does Dropout work during training? Pros/cons and interaction with Batch Norm.[Reread](3_DeepLearning.md#dropout)
    
43. List the key parameters for Early Stopping.[Reread](3_DeepLearning.md#early-stopping)
    
44. What is Batch Normalization? Write the z-score formula and the final output with \( \gamma \) and \( \beta \). Placement in layers?[Reread](3_DeepLearning.md#batch-norm)
    
45. What is Data Augmentation in the context of regularization? (Brief recall.)[Reread](3_DeepLearning.md#data-augmentation)

### Optimizer
46. Write the basic SGD update formula, and the version with Momentum (including \( \mu \)).[Reread](3_DeepLearning.md#sgd)
    
47. What does RMSProp do? Key formulas for \( E[g^2]_t \) and the update, and why it adapts per parameter.[Reread](3_DeepLearning.md#rmsprop)
    
48. Explain Adam: the two moments it estimates, bias correction formulas, and the final update. Why square the second moment?[Reread](3_DeepLearning.md#adam)
    
49. How does AdamW differ from Adam, especially with L2 regularization?[Reread](3_DeepLearning.md#adamw)
    
50. What is AdaGrad, and its relation to RMSProp?[Reread](3_DeepLearning.md#ada-grad)

## Architectures

### Convolutional Neural Networks
51. What is the main purpose of Convolutional Neural Networks, and what type of data do they process?[Reread](3_DeepLearning.md#convolutional-neural-networks)
    
52. What is a Kernel/Filter in CNNs, and how does it extract features?[Reread](3_DeepLearning.md#working)
    
53. For Convolution layers: define Filter Size F, Stride S, Padding P (types), and output size formula.[Reread](3_DeepLearning.md#convolution)
    
54. Compare Max Pooling and Average Pooling: pros and what they preserve.[Reread](3_DeepLearning.md#pooling)
    
55. What are Conv Transpose and Unpooling used for?[Reread](3_DeepLearning.md#conv-transpose) | [Reread](3_DeepLearning.md#unpooling)
    
56. Why are CNNs still used despite not being SOTA, compared to ViT?[Reread](3_DeepLearning.md#usage)
    
57. Name two main applications of CNNs.[Reread](3_DeepLearning.md#applications)
    
58. What two topics are suggested "To Explore" for CNNs?[Reread](3_DeepLearning.md#to-explore)

### Residual Connections / Skip Connections
59. What are Residual (Skip) Connections, and their formula \( H(x) \)? Main benefit?[Reread](3_DeepLearning.md#residual-connections--skip-connections)
    
60. Name two notable implementations of Residual Connections. Describe U-Net's structure briefly.[Reread](3_DeepLearning.md#notable-implementations)

### Recurrent Neural Networks
61. How do RNNs handle sequences? Describe hidden state formula and training method (BPTT).[Reread](3_DeepLearning.md#recurrent-neural-networks)
    
62. For Vanilla RNN: write the hidden state and output formulas.[Reread](3_DeepLearning.md#vanilla-rnn)
    
63. What are the three gates in LSTM, and write the key cell state update \( c_t \). Role of C vs. H?[Reread](3_DeepLearning.md#lstm)
    
64. How many gates does GRU have compared to LSTM?[Reread](3_DeepLearning.md#gru)

### Encoder Decoder
65. What are the three main shortcomings of basic Encoder-Decoder architectures?[Reread](3_DeepLearning.md#shortcomings)

### Attention Mechanism
66. Where is Attention Mechanism explained in detail? (Recall the link.)[Reread](4_Transformers.md#attention-mechanism)

### Transformer
67. Where is Transformer explained?[Reread](4_Transformers.md#transformer)

### Generative Adversarial Network
68. What is a GAN? (Placeholder recall.)[Reread](3_DeepLearning.md#generative-adversarial-network)

### Auto Encoder
69. What is an Auto Encoder?[Reread](3_DeepLearning.md#auto-encoder)

### Variational Auto Encoder
70. What is a Variational Auto Encoder?[Reread](3_DeepLearning.md#variational-auto-encoder)

### Diffusion Networks
71. What are Diffusion Networks?[Reread](3_DeepLearning.md#diffusion-networks)

### Transfer Learning
72. Define Transfer Learning and its common technique (pretrained model reuse).[Reread](3_DeepLearning.md#transfer-learning)
    
73. What is a Feature Extractor in Transfer Learning, and its use in perceptual loss?[Reread](3_DeepLearning.md#common-uses-tl)
    
74. Compare Freezing and Fine-Tuning in Transfer Learning: when to use each?[Reread](3_DeepLearning.md#how-to-implement)

### Training Strategies
75. What is Teacher Forcing, its benefits, and the "Exposure Bias" problem?[Reread](3_DeepLearning.md#teacher-forcing)
    
76. How does Scheduled Sampling solve Exposure Bias, and its drawbacks?[Reread](3_DeepLearning.md#scheduled-sampling)
    
77. Describe Curriculum Learning: two styles, pros/cons.[Reread](3_DeepLearning.md#curriculum-learning)
    
78. What is Professor Forcing, and how does it relate to GANs? Pros/cons.[Reread](3_DeepLearning.md#professor-forcing)
    
79. What is Label Smoothing? (Brief recall from context.)[Reread](3_DeepLearning.md#label-smoothing)