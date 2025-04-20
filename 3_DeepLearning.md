
  

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
	1. Temp: The effect of vanishing gradients is that gradients from time steps that are far away do not contribute anything to the learning process, so the RNN ends up not learning any long-range dependencies
11. **Exploding Gradient**
12. **How to diagnose and fix both gradient issues**
	1. Temp: Exploding gradients can be controlled by clipping them at a predefined threshold. TensorFlow 2.0 allows you to clip gradients using the clipvalue or clipnorm parameter during optimizer construction, or by explicitly clipping gradients using tf.clip_by_value
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
	
	2. Backpropagation through time (BPTT)
	Just like traditional neural networks, training RNNs also involves the backpropagation of gradients. The difference, in this case, is that since the weights are shared by all time steps, the gradient at each output depends not only on the current time step but also on the previous ones. This process is called backpropagation through time.
	
	`⚠️[Requires Investigation]` Check page: https://dennybritz.com/posts/wildml/recurrent-neural-networks-tutorial-part-3/

	3. BPTT Vulnerable to exploding and vanishing gradients
		BPTT is just regular backpropagation but unrolled through time:
		- The RNN processes a sequence step by step, updating its hidden state at each time step.
		- During training, the loss from the final time step is backpropagated through all previous time steps, using the chain rule.
		- This means gradients are multiplied repeatedly, one time step after another.

		Two Bad Scenarios:
		- Vanishing Gradients: If the spectral norm (max singular value) of W_h is < 1, the gradients shrink over time.	After enough time steps, they’re so small they’re practically zero → no learning for earlier layers.
		Especially bad with sigmoid and tanh, because their derivatives are small for large inputs.
		- Exploding Gradients:	If W_h has a norm > 1, each multiplication makes the gradient grow exponentially. Eventually the gradients blow up → unstable training, NaNs, etc.
	3. Solution to exploding and vanishing `ℹ️[Mentioned in how to diagnose and fix both gradients]`
		- Of the two, exploding gradients are more easily detectable. The gradients will become very large and turn into Not a Number (NaN), and the training process will crash . Exploding gradients can be controlled by clipping them at a predefined threshold.

		A few approaches exist towards minimizing the problem, such as proper initialization of the W matrix, more aggressive regularization, using ReLU instead of tanh activation, and pretraining the layers using unsupervised methods, the most popular solution is to use LSTM or GRU architectures.
	4. Variants
		1. **Long short-term memory (LSTM)**

		![](/Images/3_deepLearning_lstm_1.png)
		The set of equations representing an LSTM is shown as follows:

		![](/Images/3_deepLearning_lstm_2.png)

		Here, i, f, and o are the input, forget, and output gates. They are computed using the same equations but with different parameter matrices W<sub>i</sub>, U<sub>i</sub>, W<sub>f</sub>, U<sub>f</sub>, and W<sub>o</sub>, U<sub>o</sub>. The sigmoid function modulates the output of these gates between 0 and 1, so the output vectors produced can be multiplied element-wise with another vector to define how much of the second vector can pass through the first one.

		2.
16. **Normalization** `ℹ️[Mentioned in Data Processing]`
17. 