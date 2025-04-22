
  

# Machine Learning Topics

Index of notations to complete/learn more:
`⚠️[Requires Investigation]`
`❌[Incomplete]`


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
		<br/><br/>
		2.	Why is Relu still more prevalent despite leaky relu problem?
		<br/><br/>
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
	1. Cause
	As you go backward through a deep network (from output toward the input layer), gradients are calculated via the chain rule. That means:
	$$\frac{dL}{dx} = \frac{dL}{dz_n} \cdot \frac{dz_n}{dz_{n-1}} \cdot \ldots \cdot \frac{dz_2}{dz_1}$$

	If each of those derivatives is a number less than 1 (like 0.5), and you multiply a bunch of them together… the product shrinks exponentially. Eventually the gradient becomes so small that it’s practically zero.
	When that happens:
	-	Weights stop updating
	-	Neurons stop learning
	-	Your model gets stuck
	-	Early layers (closer to the input) get almost no gradient signal

	2. How to detect
	Methods To Solve:
	✅ Use ReLU instead of sigmoid/tanh
	ReLU’s derivative is 1 for positive values — no shrinking
	✅ Batch Normalization
	Helps keep the activations and gradients in a healthy range
	✅ Residual Connections (ResNets)
	Skip connections help gradients flow more easily through deep networks
	✅ Careful weight initialization
	`⚠️[Requires Investigation]` Methods like He or Xavier initialization aim to preserve the scale of activations and gradients

	3. How to Solve


	1. Temp: The effect of vanishing gradients is that gradients from time steps that are far away do not contribute anything to the learning process, so the RNN ends up not learning any long-range dependencies

11. **Exploding Gradient**`❌[Incomplete]`
	Exploding gradients occur when the gradients (partial derivatives of the loss with respect to the model parameters) become very large during backpropagation.
	These large values can cause:
	- Model weights to grow excessively. 
	- Training to become unstable.
	- Loss to oscillate wildly or become NaN.

	1. Cause
	2. How to Detect
	3. How to Fix
		1.	Gradient Clipping
		Cap the gradients to a maximum value to prevent them from getting too large.
		python
		CopyEdit
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		2.	Weight Regularization
		Techniques like L2 regularization help keep weights small.
		3.	Use Better Activation Functions
		Like ReLU or variants that are less prone to derivative explosion than, say, tanh or sigmoid.
		4.	Careful Initialization
		Initialize weights with methods like Xavier or He initialization to avoid large initial gradients.
		5.	Use Residual Connections
		Especially in very deep networks (e.g., ResNets), these help with gradient flow.

12. **How to diagnose and fix both gradient issues**`❌[Incomplete]`
	1. Temp: Exploding gradients can be controlled by clipping them at a predefined threshold. TensorFlow 2.0 allows you to clip gradients using the clipvalue or clipnorm parameter during optimizer construction, or by explicitly clipping gradients using tf.clip_by_value
13. **Convolutional Neural Networks**
	1. Definition
	Convolutional Neural Networks (CNNs) are a type of deep learning model specifically designed for processing structured grid-like data, such as images.
	2. Working
	These layers apply filters (kernels) to input images to extract important features like edges, textures, and patterns.
	Each filter slides over the input (convolution operation), producing a feature map.
	The pooling layers are used to reduce the spatial size of feature maps while retaining important information
	After extracting features, the output is flattened and passed through dense layers.

	The **kernel** is just a small matrix (e.g., 3×3 or 5×5) that slides over the input image (This kernel is trained as goal of this process).
	It performs a convolution operation by multiplying its values with the pixel values of the input and summing them up. This results in a new matrix called a **feature map**.

	Pooling is not a learnable operation—it’s just a way to reduce the size of the feature map.
	It takes the feature map (produced by convolution) and applies an operation like:
	-	Max Pooling → Takes the maximum value in a small window (e.g., 2×2).
	-	Average Pooling → Takes the average of values in a small window.

	3. Need
	Images have a spatial structure (e.g., pixels in a face are related to nearby pixels). Fully connected layers treat all pixels as independent, losing important spatial context. Filters detect edges, textures, and shapes locally and pass them deeper into the network.
	
	4. Applications:
	-	Style Transfer
	
	5.	Components
		1.	Convolution
			1.	Different Types:
				| Layer Type         | Description                                                        | Input Type                                         |
				|--------------------|--------------------------------------------------------------------|----------------------------------------------------|
				| Conv1D             | Used for time series, audio, or NLP tasks                         | 1D sequences (e.g., speech, text embeddings)       |
				| Conv2D             | Used for image processing                                          | 2D data (e.g., grayscale/RGB images)               |
				| Conv3D             | Used for volumetric data like medical imaging or videos           | 3D data (e.g., MRI scans, video frames)            |
				| Conv2DTranspose    | Used for upsampling (e.g., image segmentation, GANs)              | 2D data, like Conv2D but increases spatial size    |
				| Conv3DTranspose    | Used for 3D upsampling, such as in medical image reconstruction   | 3D data, like Conv3D but increases spatial size    |

		2.	Params
		-	Kernel: eg. (5,5) specifies the kernel size used
		-	Padding: Refers to adding extra pixels around the input image before applying the convolution operation. This is done to control the spatial size (height & width) of the output feature map.
			Types:
			1.	Valid: No padding
			2.	Same: Zero Padding (output size = input size)
		-	Number of filters: Number of different feature maps generated

		3.	Pooling`❌[Incomplete]`
		4.	Padding`❌[Incomplete]`
	
	6. Historical Performance
	

14. **Residual Connections**
	Residual connections (also called skip connections) are a technique introduced in ResNet (Residual Networks) that help train very deep neural networks by allowing the network to "skip" one or more layers.

	In very deep networks:
	-	Training gets harder due to vanishing/exploding gradients
	-	The network may start performing worse as depth increases (which is counterintuitive). Residual connections allow the network to learn residuals—that is, how much to change the input rather than learning the full transformation from scratch. This helps:
	-	Improve gradient flow (prevent vanishing/exploding gradients)
	-	Make optimization easier
	-	Enable successful training of networks with 100+ layers

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

		LSTM is a drop-in replacement for a SimpleRNN cell.
		LSTMs are resistant to the vanishing gradient problem.

		2.**Gated recurrent unit (GRU)**
		GRU is a variant of the LSTM.retains the LSTM’s resistance to the vanishing gradient problem, but its internal structure is simpler, and is, therefore, faster to train, since fewer computations are needed to make updates to its hidden state.
		Instead of the input (i), forgot (f), and output (o) gates in the LSTM cell, the GRU cell has two gates, an update gate z and a reset gate r. The update gate defines how much previous memory to keep around, and the reset gate defines how to combine the new input with the previous memory. There is no persistent cell state distinct from the hidden state as it is in LSTM.

		![](/Images/3_deepLearning_gru_1.png)

		The outputs of the update gate z and the reset gate r are both computed using a combination of the previous hidden state h<sub>t-1</sub> and the current input x<sub>t</sub>.
		The sigmoid function modulates the output of these functions between 0 and 1. The cell state c is computed as a function of the output of the reset gate r and input xt. Finally, the hidden state ht at time t is computed as a function of the cell state c and the previous hidden state ht-1. The parameters Wz, Uz, Wr, Ur, and Wc, Uc, are earned during training.

		3. **Peephole LSTM**
		4. **Bidirectional RNNs**
		5. **Stateful RNNs**
	5. **Topologies**
		![](/Images/3_deepLearning_rnn_5.png)

		Many-to-many use case comes in two flavors. The first one is more popular and is better known as the seq2seq model. In this model, a sequence is read in and produces a context vector representing the input sequence, which is used to generate the output sequence.

		Second many-to-many type has an output cell corresponding to each input cell. This kind of network is suited for use cases where there is a 1:1 correspondence between the input and output, such as time series. The major difference between this model and the seq2seq model is that the input does not have to be completely encoded before the decoding process begins.
	6. 
		

16. **Normalization** `ℹ️[Mentioned in Data Processing]`
17. 