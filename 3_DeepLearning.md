
  

# Machine Learning Topics

## Deep Learning
1.  **Perceptron**
	A Perceptron is the simplest type of artificial neural network. It is a type of linear classifier that makes predictions based on a weighted sum of input features followed by an activation function. 
	Composed of: A perceptron consists of the following components: 
	1. **Inputs** (x1,x2,...,xnx_1, x_2, ..., x_nx1,x2,...,xn): Features of the data. 
	2. **Weights** (w1,w2,...,wnw_1, w_2, ..., w_nw1,w2,...,wn): Adjustable parameters that determine the importance of each feature. 
	3. **Bias** (b): A constant term that allows shifting the decision boundary. 
	4. **Summation Function:** Computes the weighted sum of inputs

		![](/Images/3_deepLearning_1_1.png)
	5. **Activation Function**: Applies a step function (threshold function) to determine the output:
	
		![](/Images/3_deepLearning_1_2.png)
3.  **Feed Forward Neural Networks**
4.  **Hidden Layers**
5.  **Width of Model**
6. **Weights and Biases**
7.  **Universal Approximation Theorem**
8.  **Activation Function**
9. **Gradient Descent**
	1. Stochastic
	2. Batch
	3. Mini Batch
10.  **Vanishing Gradient**
11. **Exploding Gradient**
12. **How to diagnose and fix both gradient issues**
13. **Convolutional Neural Networks**
14. **Residual Connections**
15. **Recurrent Neural Networks**
	1. **Working**
	Used for sequences where the value of last element does predict next. The value of the hidden state at any point in time is a function of the value of the hidden state at the previous time step, and the value of the input at the current time step.

	![](/Images/3_deepLearning_rnn_1.png)

	![](/Images/3_deepLearning_rnn_2.png)
	
The output vector y<sub>t</sub> at time _t_ is the product of the weight matrix _V_ and the hidden state h<sub>t</sub>, passed
through a SoftMax activation, such that the resulting vector is a set of output probabilities.

	![](/Images/3_deepLearning_rnn_3.png)
	
	2. 
16. **Normalization** `ℹ️[Mentioned in Data Processing]`
17. 