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