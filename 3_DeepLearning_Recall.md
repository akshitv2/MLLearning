# Deep Learning – Recall Questions
## Perceptron

1. What are the components of a perceptron?
2. Write the equation for the weighted sum in a perceptron.
3. What activation function does a perceptron use, and what is its output rule?
4. Feed Forward Networks & Hidden Layers
5. What is a feedforward neural network?
6. What are hidden layers in a neural network?
7. What does “width of model” refer to?
8. Universal Approximation Theorem
9. What does the Universal Approximation Theorem state?
10. Does it guarantee that backpropagation can find the mapping?

## Activation Functions

12. Why are activation functions needed?
13. Write the mathematical form of ReLU. What is its main drawback?
14. What problem does Leaky ReLU solve and how?
15. What is the difference between Leaky ReLU and Parametric ReLU?
16. What is the range of the sigmoid function, and why is it not used much?
17. What is the range of tanh? What problem does it suffer from?
18. What does softmax output sum to, and why do we exponentiate z values?
19. Compare Sigmoid vs Softmax in terms of use cases.
20. Why is ReLU more prevalent than Leaky ReLU despite its problems?

## Gradient Descent

22. Write the update rule for gradient descent.
23. What is the main difference between stochastic, batch, and mini-batch gradient descent?
24. What are pros and cons of SGD compared to batch gradient descent?

## Epoch

26. What does one epoch mean in training?
27. Vanishing & Exploding Gradients
28. Why do gradients vanish when propagating backward?
29. What are signs of vanishing gradients?
30. Name three methods to reduce vanishing gradients.
31. What happens when gradients explode?
32. How can exploding gradients be fixed?

## CNNs

34. What kind of data are CNNs designed for?
35. What does a kernel do in CNNs?
36. What is the difference between convolution and pooling in terms of learnability?
37. Give an example of max pooling vs average pooling.
38. Why are CNNs preferred over dense layers for images?
39. What are Conv1D, Conv2D, Conv3D used for?
40. What does “padding” do in convolutions?
41. Which models historically achieved the highest top-1 accuracy in the CNN performance table?
42. Residual Connections
43. Why were residual connections introduced?
44. How do they help with vanishing/exploding gradients?

## RNNs

46. Write the formula for updating hidden state in an RNN.
47. What is backpropagation through time (BPTT)?
48. Why are RNNs prone to vanishing/exploding gradients?
49. What are LSTM gates and their purposes?
50. How is the forget gate initialized to avoid vanishing gradients?
51. What is the key structural difference between GRU and LSTM?
52. Encoder–Decoder (Seq2Seq)
53. What is the encoded vector in an encoder–decoder model?
54. What role does the decoder’s START token play?
55. What is teacher forcing, and what is its drawback?
56. Why does reversing input sequences help seq2seq models?
57. Attention Mechanism
58. Why is attention needed in seq2seq models?
59. How is the attention score calculated? Name two methods.
60. At what stage (encoding or decoding) are attention scores computed?

## Transformers

62. What function do positional encodings serve?
63. Write the formula for sinusoidal positional encoding.
64. How do transformers handle sequential data without recurrence?
65. What is masking used for in transformers? Name two types.
66. Why not use raw indices instead of positional encodings?
67. Why divide by √d<sub>k</sub> in scaled dot-product attention?

## Normalization

69. What problem does batch normalization address?
70. Write the normalization formula with mean and variance.
71. Why are γ and β parameters included in batch norm?
72. Regularization
73. What is the purpose of regularization?
74. How does L1 differ from L2 in effect on weights?
75. Why does L1 lead to sparsity?
76. What is early stopping, and what are its pros/cons?

## Optimizers

78. What is the role of an optimizer in training?
79. What is the main idea of RMSProp?
80. What is the main idea of Adam optimizer?
81. Why does AdaGrad’s learning rate shrink over time?
82. Which optimizer is most commonly used by default today?
    Transfer Learning
84. What is transfer learning?
85. What are the two main parts of a pretrained model?
86. What is the difference between feature extraction and fine-tuning?
87. When would you freeze vs fine-tune pretrained layers?


### Perceptron  
1. What are the components of a perceptron? [goto](3_DeepLearning.md#perceptron)  
2. Write the equation for the weighted sum in a perceptron. [goto](3_DeepLearning.md#perceptron)  
3. What activation function does a perceptron use, and what is its output rule? [goto](3_DeepLearning.md#perceptron)  

### Feed Forward Networks & Hidden Layers  
4. What is a feedforward neural network? [goto](3_DeepLearning.md#feed-forward-neural-networks)  
5. What are hidden layers in a neural network? [goto](3_DeepLearning.md#hidden-layers)  
6. What does “width of model” refer to? [goto](3_DeepLearning.md#width-of-model)  

### Universal Approximation Theorem  
7. What does the Universal Approximation Theorem state? [goto](3_DeepLearning.md#universal-approximation-theorem)  
8. Does it guarantee that backpropagation can find the mapping? [goto](3_DeepLearning.md#universal-approximation-theorem)  

### Activation Functions  
9. Why are activation functions needed? [goto](3_DeepLearning.md#activation-function)  
10. Write the mathematical form of ReLU. What is its main drawback? [goto](3_DeepLearning.md#activation-function)  
11. What problem does Leaky ReLU solve and how? [goto](3_DeepLearning.md#activation-function)  
12. What is the difference between Leaky ReLU and Parametric ReLU? [goto](3_DeepLearning.md#activation-function)  
13. What is the range of the sigmoid function, and why is it not used much? [goto](3_DeepLearning.md#activation-function)  
14. What is the range of tanh? What problem does it suffer from? [goto](3_DeepLearning.md#activation-function)  
15. What does softmax output sum to, and why do we exponentiate z values? [goto](3_DeepLearning.md#activation-function)  
16. Compare Sigmoid vs Softmax in terms of use cases. [goto](3_DeepLearning.md#activation-function)  
17. Why is ReLU more prevalent than Leaky ReLU despite its problems? [goto](3_DeepLearning.md#activation-function)  

### Gradient Descent  
18. Write the update rule for gradient descent. [goto](3_DeepLearning.md#gradient-descent)  
19. What is the main difference between stochastic, batch, and mini-batch gradient descent? [goto](3_DeepLearning.md#gradient-descent)  
20. What are pros and cons of SGD compared to batch gradient descent? [goto](3_DeepLearning.md#gradient-descent)  

### Epoch  
21. What does one epoch mean in training? [goto](3_DeepLearning.md#epoch)  

### Vanishing & Exploding Gradients  
22. Why do gradients vanish when propagating backward? [goto](3_DeepLearning.md#vanishing-gradient)  
23. What are signs of vanishing gradients? [goto](3_DeepLearning.md#vanishing-gradient)  
24. Name three methods to reduce vanishing gradients. [goto](3_DeepLearning.md#vanishing-gradient)  
25. What happens when gradients explode? [goto](3_DeepLearning.md#exploding-gradient)  
26. How can exploding gradients be fixed? [goto](3_DeepLearning.md#exploding-gradient)  

### CNNs  
27. What kind of data are CNNs designed for? [goto](3_DeepLearning.md#convolutional-neural-networks)  
28. What does a kernel do in CNNs? [goto](3_DeepLearning.md#convolutional-neural-networks)  
29. What is the difference between convolution and pooling in terms of learnability? [goto](3_DeepLearning.md#convolutional-neural-networks)  
30. Give an example of max pooling vs average pooling. [goto](3_DeepLearning.md#convolutional-neural-networks)  
31. Why are CNNs preferred over dense layers for images? [goto](3_DeepLearning.md#convolutional-neural-networks)  
32. What are Conv1D, Conv2D, Conv3D used for? [goto](3_DeepLearning.md#convolutional-neural-networks)  
33. What does “padding” do in convolutions? [goto](3_DeepLearning.md#convolutional-neural-networks)  
34. Which models historically achieved the highest top-1 accuracy in the CNN performance table? [goto](3_DeepLearning.md#convolutional-neural-networks)  

### Residual Connections  
35. Why were residual connections introduced? [goto](3_DeepLearning.md#residual-connections)  
36. How do they help with vanishing/exploding gradients? [goto](3_DeepLearning.md#residual-connections)  

### RNNs  
37. Write the formula for updating hidden state in an RNN. [goto](3_DeepLearning.md#recurrent-neural-networks)  
38. What is backpropagation through time (BPTT)? [goto](3_DeepLearning.md#recurrent-neural-networks)  
39. Why are RNNs prone to vanishing/exploding gradients? [goto](3_DeepLearning.md#recurrent-neural-networks)  
40. What are LSTM gates and their purposes? [goto](3_DeepLearning.md#recurrent-neural-networks)  
41. How is the forget gate initialized to avoid vanishing gradients? [goto](3_DeepLearning.md#recurrent-neural-networks)  
42. What is the key structural difference between GRU and LSTM? [goto](3_DeepLearning.md#recurrent-neural-networks)  

### Encoder–Decoder (Seq2Seq)  
43. What is the encoded vector in an encoder–decoder model? [goto](3_DeepLearning.md#encoder-decoder-seq2seq)  
44. What role does the decoder’s START token play? [goto](3_DeepLearning.md#encoder-decoder-seq2seq)  
45. What is teacher forcing, and what is its drawback? [goto](3_DeepLearning.md#encoder-decoder-seq2seq)  
46. Why does reversing input sequences help seq2seq models? [goto](3_DeepLearning.md#encoder-decoder-seq2seq)  

### Attention Mechanism  
47. Why is attention needed in seq2seq models? [goto](3_DeepLearning.md#attention-mechanism)  
48. How is the attention score calculated? Name two methods. [goto](3_DeepLearning.md#attention-mechanism)  
49. At what stage (encoding or decoding) are attention scores computed? [goto](3_DeepLearning.md#attention-mechanism)  

### Transformers  
50. What function do positional encodings serve? [goto](3_DeepLearning.md#transformers)  
51. Write the formula for sinusoidal positional encoding. [goto](3_DeepLearning.md#transformers)  
52. How do transformers handle sequential data without recurrence? [goto](3_DeepLearning.md#transformers)  
53. What is masking used for in transformers? Name two types. [goto](3_DeepLearning.md#transformers)  
54. Why not use raw indices instead of positional encodings? [goto](3_DeepLearning.md#transformers)  
55. Why divide by √d<sub>k</sub> in scaled dot-product attention? [goto](3_DeepLearning.md#transformers)  

### Normalization  
56. What problem does batch normalization address? [goto](3_DeepLearning.md#normalization)  
57. Write the normalization formula with mean and variance. [goto](3_DeepLearning.md#normalization)  
58. Why are γ and β parameters included in batch norm? [goto](3_DeepLearning.md#normalization)  

### Regularization  
59. What is the purpose of regularization? [goto](3_DeepLearning.md#regularization)  
60. How does L1 differ from L2 in effect on weights? [goto](3_DeepLearning.md#regularization)  
61. Why does L1 lead to sparsity? [goto](3_DeepLearning.md#regularization)  
62. What is early stopping, and what are its pros/cons? [goto](3_DeepLearning.md#regularization)  

### Optimizers  
63. What is the role of an optimizer in training? [goto](3_DeepLearning.md#optimizers)  
64. What is the main idea of RMSProp? [goto](3_DeepLearning.md#optimizers)  
65. What is the main idea of Adam optimizer? [goto](3_DeepLearning.md#optimizers)  
66. Why does AdaGrad’s learning rate shrink over time? [goto](3_DeepLearning.md#optimizers)  
67. Which optimizer is most commonly used by default today? [goto](3_DeepLearning.md#optimizers)  

### Transfer Learning  
68. What is transfer learning? [goto](3_DeepLearning.md#transfer-learning)  
69. What are the two main parts of a pretrained model? [goto](3_DeepLearning.md#transfer-learning)  
70. What is the difference between feature extraction and fine-tuning? [goto](3_DeepLearning.md#transfer-learning)  
71. When would you freeze vs fine-tune pretrained layers? [goto](3_DeepLearning.md#transfer-learning)  
