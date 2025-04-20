
  

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
	Layers between the input and output layers
5.  **Width of Model**
6. **Weights and Biases**
7.  **Universal Approximation Theorem**
	The Universal Approximation Theorem is a pivotal result in neural network theory, proving that feedforward neural networks can approximate any continuous function under certain conditions.
8.  **Activation Function**
	1.	Need
	Activation functions are crucial in neural networks because they introduce non-linearity into the model, enabling it to learn complex patterns and relationships in data. Without activation functions, a neural network would essentially be a linear model, limiting its ability to handle complex tasks.
	If we had no activation function, the output of a layer would be simply a weighted sum of the inputs z=w<sub>1</sub>x<sub>1</sub>+w<sub>2</sub>x<sub>2</sub>+...+w<sub>n</sub>x<sub>n</sub>+bz = w_1x_1 + w_2x_2 + ... + w_nx_n + bz=w1x1+w2x2+...+wnxn+b

	2.	Common ones:
		1.	**ReLU**
		Rectified Linear Unit – f(x)=max(0,x)

		![](/Images/3_deepLearning_relu_1.png)

		![](/Images/3_deepLearning_relu_2.png)

		-	Pros:
			-	Only negatively saturates
			-	Better Sparsity so less computation
		-	Cons:
			- Dying RELU (Can get stuck at 0)
			-	Not differentiable at 0 (solved using f′(0)=0)
		

		2. **Leaky ReLU** 

		![](/Images/3_deepLearning_leaky_relu_1.png)
		
		Where α: a small positive constant (usually something like 0.01).
		designed to fix a problem known as the "dying ReLU" problem
		Leaky ReLU doesn’t just cut off all negative values — instead, it lets a small negative slope through.
		So even when x<0, the function still outputs a small (negative) value and, more importantly, has a non-zero gradient.
		-	The neuron still gets to learn (because there's still a gradient to flow back during backpropagation).
		-	It reduces the risk of neurons getting “stuck” outputting 0 forever.

		3.	**Parametric ReLU**

		![](/Images/3_deepLearning_leaky_relu_1.png)

		Here alpha is not fixed and learned during training. Can be shared or different alpha per layer.
		Gives the network freedom to learn better slopes but this may cause overfitting if you're not careful.
		Slightly slower than regular Relu due to the extra computation.

		4.	**Sigmoid (Logit)**

		![](/Images/3_deepLearning_sigmoid_1.png)
		![](/Images/3_deepLearning_sigmoid_2.png)

		When x→−∞ f(x)→0
		When x→+∞ f(x)→1
		At x=0, f(x)=0.5

			Not used much anymore due to vanishing gradients (since derivative is close to 0).
		Also computationally expensive.

		5.	**Tanh**

		![](/Images/3_deepLearning_tanh_1.png)

		like sigmoid, tanh suffers from the vanishing gradient problem for very large or very small inputs and unpopular compared to RELU.
		
		6.	**Softmax**

		![](/Images/3_deepLearning_softmax_1.png)

		The softmax function takes a vector of raw scores (called logits) and turns them into probabilities.
		`❌[Incomplete]`

		7.	**Swish**

		8.	**GELU**

	<br/><br/>
	3.	Questions:
		1.	Why Non-Linearity Important?
		
		2.	Why is Relu still more prevalent despite leaky relu problem?
		
		3.	Sigmoid vs Softmax		

		|Feature     |      Sigmoid             |	               Softmax                            |
		|------------|--------------------------|-----------------------------------------------------|
		|Use Case    |Binary Classification     |Multi Class Classification                           |
		|Independence|Each output is independent|Outputs are interdependent (probability distribution)|
		|Range	     |(0, 1) for each class     |(0, 1) for each class but all sum up to 1            |

•	Sigmoid treats each class independently, meaning probabilities don’t sum to 1.
•	It can assign high probabilities to multiple classes at the same time, which is problematic when only one class should be selected.
•	Softmax ensures a mutually exclusive decision by normalizing across all classes.
•	Sigmoid is better than softmax in two main cases: Binary Classification & Multi-Label Classification of Independent classes
•	Softmax is computationally more expensive than sigmoid, especially as the number of classes increases.

9. **Gradient Descent**
	1. Stochastic
	2. Batch
	3. Mini Batch
10.  **Vanishing Gradient**`❌[Incomplete]`
	1. Temp: The effect of vanishing gradients is that gradients from time steps that are far away do not contribute anything to the learning process, so the RNN ends up not learning any long-range dependencies
11. **Exploding Gradient**
12. **How to diagnose and fix both gradient issues**`❌[Incomplete]`
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

		The forget gate defines how much of the previous state h<sub>t-1</sub> you want to allow to pass through. The input gate defines how much of the newly computed state for the current input x<sub>t</sub> you want to let through, and the output gate defines how much of the internal state you want to expose to the next layer. The internal hidden state g is computed based on the current input x<sub>t</sub> and the previous hidden state h<sub>t-1</sub>

		2.
16. **Normalization** `ℹ️[Mentioned in Data Processing]`
17. 