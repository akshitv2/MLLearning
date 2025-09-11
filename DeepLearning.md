# Deep Learning

## Basics

1. **Perceptron**
   Simplest type of artificial neural network.  
   Makes prediction based on a single input by w*x+b followed an activation function.
    1. **Inputs** (x1,x2,...,xnx_1, x_2, ..., x_nx1,x2,...,xn): Features of the data.
    2. **Weights** (w1,w2,...,wnw_1, w_2, ..., w_nw1,w2,...,wn): Adjustable parameters that determine the importance of
       each feature.
    3. **Bias** (b): A constant term that allows shifting the decision boundary.
    4. **Summation Function:** Computes the weighted sum of inputs
       $$z = \sum_{i=1}^{n} w_i x_i + b$$
    5. **Activation Function**: Applies a step function (threshold function) to determine the output:
       $$y = \begin{cases} 1, & \text{if } z \geq 0 \\ 0, & \text{otherwise} \end{cas
2. **Feed Forward Neural Networks**
3. **Hidden Layers**<br>
   Layers between the input and output layers
4. ### Width of Model**
5. ### Weights and Biases**
   (Mentioned above)
   Weights: Defined for each connection. Variable input is multiplied with.  
   Biases: Defined for each node. Variable input shifted by.
6. ### Universal Approximation Theorem
   Pivotal theorem, proving that provided a sufficiently deep neural network with non linear activation can approximate
   any function. (not a proof of finding it but least knowing that it's possible).
7. ### Activation Function:
   Function applied to the output of a neural network.
    1. #### Need:
       Introduce Non Linearity: Without them NN of any depth would be same as one linear transformation
    2. #### Types:
        1. ##### Sigmoid [DEPRECATED]
           $$f(x) = \frac{1}{1 + e^{-x}}$$  
           <img src="images/img_4.png" alt="img" width="300">
            - as x→∞ y-> 1
            - as x->-∞ y-> 0
            - at x = 0, y = 0.5  
              Pro Cons:
            - 🔴 Obsolete
            - 🔴 f`(x) maxes out at 0.25 i.e sure to cause vanishing gradient as you add more
        2. ##### Tanh [DEPRECATED]
           $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
            - as x→∞ y-> 1
            - as x->-∞ y-> -1
            - at x = 0, y = 0  
              <img src="images/img_3.png" alt="img" width="300">
            - Pro Cons:
                - 🔴 Obsolete
                - 🔴 f`(x) maxes out at 1 i.e sure to cause vanishing gradient as you add more
        3. ##### ReLU (Rectified Linear Unit)
           $$f(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$
            - 🟢 Only negatively saturates
            - 🟢 Better Sparsity so less computation
            - 🔴 Dying RELU (Can get stuck at 0)
            - 🔴 Not differentiable at 0 (solved using f′(0)=0)
        4. ##### Leaky ReLU
        5. ##### Parametric ReLU
        6. ##### Swish
        7. ##### eLU
        8. ##### Softmax

8. ### Gradient Descent
    1. Types:
        1. ### Stochastic
        2. ### Batch
        3. ### Minibatch
        4. ### Momentum Based
        5. ### Nesterov Accelerated Gradient Descent
    2. ### Common Issues:
        1. Vanishing Gradient
        2. Exploding Gradient
9. ### Weight Initialization
    1. #### Zero Init
    2. #### Random Init
    3. #### Xavier Init
    4. #### He Init
10. ### Learning Rate Scheduling
     1. #### Step Decay
     2. #### Exponential Decay
     3. #### Cosine Annealing
     4. #### Polynomial Decay
     5. #### LR On Plateau
     6. #### Cyclical
11. ### Regularization
    1. #### L1 LASSO : Least Absolute Shrinkage and selection operator 
    2. #### L2 Ridge
    3. #### Elastic Net
    4. #### Dropout
    5. #### Early Stopping
    6. #### Batch Norm
    7. #### Data Augmentation?
12. ### Optimizer
    1. #### SGD
    2. #### RMSProp
    3. #### Adam
    4. #### AdamW
    5. #### Ada grad

## Architectures
1. ### Convolutional Neural Networks
2. ### Recurrent Neural Networks
   1. #### Vanilla RNN
   2. #### LSTM
   3. #### GRU
3. ### Encoder Decoder
4. ### Transformer
5. ### Generative Adversarial Network
6. ### Auto Encoder
7. ### Variational Auto Encoder
8. ### Diffusion Networks