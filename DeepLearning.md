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
   Pivotal theorem, proving that provided a sufficiently deep neural network with non-linear activation can approximate
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
           $$f(x) = \begin{cases} x, & \text{if } x \geq 0 \\ \alpha x, & \text{if } x < 0 \end{cases}$$
            - 🟢 Solves dying relu by letting a small amount of negative gradient through 0< $\alpha$ <<1
            - 🔴 Fixes dying relu but at cost of sparsity
        5. ##### Parametric ReLU
           $$f(x) = \begin{cases} x, & \text{if } x \geq 0 \\ \alpha x, & \text{if } x < 0 \end{cases}$$
            - Same equation as leaky except alpha is a learnable param
            - 🟢 Solves dying relu again
            - 🔴 At cost of sparsity and increased computation
        6. ##### Swish
           $$f(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}$$
            - $ \beta $ is usually 1
            - Behaves same as relu for x >>0
            - 🟢 provides negative gradient solving dying relu
            - 🔴 Gains are very task dependant, not a universal choice. Used only in deep CNNs
            - 🔴 Increases computation cost
        7. ##### eLU (Exponential Linear Unit)
           $$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha \left(e^x - 1\right) & \text{if } x \leq 0 \end{cases}$$
            - 🟢 smooth and allows -ve gradient
            - 🟢 centers at 0 with a smooth negative gradient
            - 🔴 more computation
            - requires $ \alpha $ tuning
        8. ##### Softmax
           $$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$
            - Used in multi class classification gives normalized probabilities which sum to 1
            - uses e because summing + and - can cancel some out and e^x is never 0
            - The softmax function takes a vector of raw scores (called logits) and turns them into probabilities.

8. ### Gradient Descent
   First order iterative algorithm to find local minima of loss function
    - Learning Rate: Determines the size of step taken
    - Epoch: One go through of entire dataset
    - Shuffling: Randomizing order of dataset before every epoch.

    1. Types:
        1. ### Stochastic
            - Uses one training example per update
            - 🔴 Noisy updates can cause zigzagging
            - 🔴 Unstable
            - 🟢 Stochasticity can help escape local minima
        2. ### Batch
            - Uses entire dataset i.e update once every epoch
            - 🟢 Very stable
            - 🔴 Very slow
            - 🔴 Consumes a lot of memory loading entire DS into memory
            - 🟢 Smooth Convergence
        3. ### Minibatch
            - Uses smaller batch sizes usually 32,64,128 and updates per mini batch
            - Good middle ground
            - 🟢 Smoother than stochastic
            - 🟢 Faster convergence than batch
            - 🔴 Requires tuning batch size
        4. ### Momentum Based
            - Not a metric of data set used.
            - Adds a fraction of previous update to accelerate descent
            - $$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta_{t-1})$$
            - $$\theta_t = \theta_{t-1} - v_t$$
            - vt is velocity
            - $ \gamma$ momentum coefficient (i.e how much past gradient matters)
            - $ \eta $ learning rate

        5. ### Nesterov Accelerated Gradient Descent
            - Adds look ahead to momentum i.e. calculates descent from a position which is already at position post this
              update
            - $$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta_{t-1} - \gamma v_{t-1})$$
            - 🟢 faster convergence
            - Optimizer is peeking ahead and adjusting course in direction before overshooting.

    2. ### Common Issues:
        1. ### Vanishing Gradient
        2. ### Exploding Gradient
9. ### Backpropagation
10. ### Weight Initialization
    1. #### Zero Init
        - Initiliaze all weights as 0
        - 🔴⚠️ Terrible idea, any time all weights have same value causes symmetric learning i.e. all neurons in layer
          learn same values
    2. #### Random Init (Naive)
        - Assigns random values to avoid zero init
        - if weights are too small or too large will cause vanishing/exploding gradient
    3. #### Xavier Init
        - designed to keep variance and gradients approx same across all layers to avoid vanish/exploding gradient
        - ideal for **tanh/sigmoid**
        - 2 types:
          -
          Uniform: $$W \sim \mathcal{U}\!\left(-\sqrt{\tfrac{6}{n_{\text{in}} + n_{\text{out}}}}, \; \sqrt{\tfrac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$
            - Normal: $$W \sim \mathcal{N}\!\left(0, \; \tfrac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$
        - n<sub>in</sub> and n<sub>out</sub> are number of connections in and out respectively
        - symmetric around 0 and squash values, so both the input side (fan-in) and output side (fan-out) matter
    4. #### He Init
        - designed to keep variance and gradients approx same across all layers to avoid vanish/exploding gradient
        - ideal for **Relu**
        - 2 Types:
          -
          Uniform: $W \sim \mathcal{U}\!\left(-\sqrt{\tfrac{6}{n_{\text{in}}}}, \; \sqrt{\tfrac{6}{n_{\text{in}}}}\right)$
            - Normal: $W \sim \mathcal{N}\!\left(0, \; \tfrac{2}{n_{\text{in}}}\right)$
        - n<sub>in</sub> are number of connections in
        - Relu halves outputs (only +ves), variance of he doubled to compensate (compared to Xavier)
        - But since the key variance-preserving step happens on the input side, He init only uses fan-in
        - (on output side dL/dz gives 1 or 0 only unlike Xavier where it's a complex value)
11. ### Learning Rate Scheduling
    1. #### Step Decay
        - $\eta_t = \eta_0 \cdot \gamma^{\left\lfloor \tfrac{t}{T} \right\rfloor}$
        - Drops LR by a constant factor every few epochs
    2. #### Exponential Decay
        - $\eta_t = \eta_0 \cdot e^{-\lambda t}$
        - Drops learning rate exponentially per epoch
        - ![img_9.png](img_9.png)
    3. #### Cosine Annealing
        - Follows gentler cosine function
        - $\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min}) \left(1 + \cos\!\left(\frac{\pi t}{T_{\max}}\right)\right)$
        - ![img_10.png](img_10.png)
    4. #### Polynomial Decay
        - $\eta_t = \eta_0 \left( 1 - \frac{t}{T_{\max}} \right)^p$
        - Constant polynomial decay
    5. #### LR On Plateau
        - Reduces when a validation metric plateaus (i.e. stops improving)
    6. #### Cyclical
        - Increases and decreases learning rate
        - 🔴 Need max and min and cycle rate careful tuning
        - 🟢 can help get out of minima
        - ![img_7.png](img_7.png)
    7. #### One cycle
        - Increases initially then decreases rapidly
        - Gives fast convergence
        - ![img_11.png](img_11.png)
12. ### Regularization
    1. #### L1 LASSO : Least Absolute Shrinkage and selection operator
        - Applies linear penalty to magnitude of weight
        - ![img.png](Images/3_deepLearning_L1_regularization.png)
        - Forms n dimensional diamond constraint region
        - ![img_13.png](img_13.png)
        - 🟢 Causes Sparsity which can speed up computation
        - 🟢 Makes model more interpretable
        - 🔴 Sparsity can force useful weights to 0, once set to 0 always vanishes
    2. #### L2 Ridge
        - ![img.png](Images/3_deepLearning_L2_regularization.png.png)
        - Applies quadratic penalty to magnitude of weight
        - Since penalty is quad doesn't force sparsity
        - Since w < 1 would make w^2 even smaller
        - And w>1 would be much larger so minimizes this first
        - 🟢 Prevents overfitting while keeping all the features
        - Forms circular constraint region
        - ![img_14.png](img_14.png)
    3. #### Elastic Net
        - ![img.png](Images/3_deepLearning_ElasticNet_regularization.png)
        - Combines L1 and L2
        - Combines benefit of both
        - 🔴 Requires tuning of relative lambda 1 and 2 for benefits
    4. #### Dropout
        - Temporarily disable output of select randomly chosen neurons while training
        - Chooses based on probablity p (hyperparam)
        - 🟢 Reduces overbalance of model on select neurons
        - 🔴 Slows convergence
        - Requires careful tuning with batch normalization (since batch norm computes mean and variance of all outputs
          while training and uses them while eval and if some are missing while training will skew the numbers)
    5. #### Early Stopping
        - Monitors performance on validation set and stops training once plateaus
        - i.e. some validation stops increasing
        - Params:
            - Patience: number of epochs
            - Monitor: Which metric to monitor
            - Restore Best Weights
            - Min_Delta: What counts as a plateau
        - 🟢 Easy to implement
        - 🟢 Reduces overfitting
        - 🔴 Requires validation set
        - 🔴 May prematurely stop (good training can have plateaus)
    6. #### Batch Norm
        - Applied after linear and convolution layer but before activation function
        - (Linear / Convolution)→BatchNorm→Activation(e.g. ReLU)
        - Converts Values to Z score:
        - $\hat{z} = \frac{z - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
            - $\epsilon$ term is a small positive number to prevent divide by 0
        - Then adds trainable $\gamma$ and $\beta$ to shift (since some models perform better when mean =/ 0)
        - $z_{\text{BN}} = \gamma \hat{z} + \beta$
        - 🟢 Faster training on regularized terms
        - 🟢 Reduces exploding and vanishing gradient
        - 🔴 Requires careful usage with dropout
    7. #### Data Augmentation?
13. ### Optimizer
    1. #### SGD
    2. #### RMSProp
    3. #### Adam
    4. #### AdamW
    5. #### Ada grad

## Architectures

1. ### Convolutional Neural Networks
   ![img_12.png](img_12.png)
    - #### Purpose:
        - For processing grid structured data like images
    - #### Working:
        - Apply kernels to input images to extract important features
        - Kernel: Smaller matrix that slides over image
    - #### Layer Types:
        - ##### Convolution:
            - Performs convolution operation with Filter of **size F** and **Stride S**
            - Filter is trainable
            - Specify n to have n different filters to produce n separate feature maps
            - Params:
                - Filter Size F
                - Stride S
                - Padding P -> Extra 0 pixels added to edges of image (in case we don't want to downsample)
                    - Valid/No -> None Added
                    - Same -> Pads to same as input
                    - Full -> Actually upsamples, each layer gets it's full conv
                    - Formula Output Size = 1+(N-F)/S where N is input dimension
        - ##### Pooling:
            - Pooling non trainable downsampling operation
            - Applies simple operation on its filter elements:
                - Max -> 🟢 Preserves Best detected features
                - Average -> 🟢 Preserves overall features
        - ##### Conv Transpose:
            - Opposite of convolution, transpose
            - Upsampling operation, trainable
        - ##### Unpooling:
            - Not as popular
            - Sets middle index and rest 0
        - ##### Fully Connected:
            - Good Ol' Fully Connected Layer
    - #### Usage:
        - Deep CNN themselves no longer SOTA but are used extensively in SOTA models e.g. UMAP in diffusion
        - 🟢 Less resource intensive than ViT
        - 🟢 Easier to train on small datasets, ViT have massive DS requirements
    - #### Applications:
        - Image Classification
        - Object Detection (can do singular and multiple as well)
    - To Explore:
        - Object Detection Loss
        - Notable networks and their architectures
2. ### Residual Connections / Skip Connections
    - NN which has shortcuts from earlier to deeper layers
    - 🟢 Helps Signal and Gradient flow through better (by ensuring signal is not lost across layers)
    - Essentially H(x)=F(x)+x
    - 🟢 Enables training of networks with hundreds of layers
    - ### Notable Implementations:
        - Almost every very deep network (from transformer to deep cnn to Diffusion)
        - Unet (Uses Residual and CNN)
            - Shaped like a U. Walls are downsampling operations and upsampling operations (conv)
            - Left Side: Encoder -> Downsamples
            - Right Side: Decoder -> Upsamples
            - Bottle Neck: Exists at deeper layers
            - Residual connections exist from each layer left to right to ensure bottleneck doesn't stop data
            - ![img_15.png](img_15.png)
3. ### Recurrent Neural Networks
    - Composed of sequential units that use previous output and have hidden states carried forward
    - Each neuron feeds into itself at every timestep, shown below unrolled
    - ![](/Images/3_deepLearning_rnn_2.png)
    - Hidden state is a function of last hidden state and input $$h_t = \phi(h_{t-1}, X_t)$$
    - Trained using backpropagation through time (BPTT) i.e. same weights are trained calculating gradient multiple
      times for each sequence.
    - Can have multiple input output configurations
    - ![](/Images/3_deepLearning_rnn_5.png)

    1. #### Vanilla RNN
        - Usually rely on these formulas
        - $$h_t = \tanh(Wh_{t-1} + Ux_t)$$
        - $$y_t = \mathrm{softmax}(Vh_t)$$
        - Softmaxxed to give resulting probablities (like in text generation)
    2. #### LSTM
        - ![img_17.png](img_17.png)
        - Equations:
            - $$i = \sigma(W_i h_{t-1} + U_i x_t + V_i c_{t-1})$$
            - $$f = \sigma(W_f h_{t-1} + U_f x_t + V_f c_{t-1})$$
            - $$o = \sigma(W_o h_{t-1} + U_o x_t + V_o c_{t-1})$$
            - $$\mathrm{\tilde{C}}_{t} = \tanh(W_g h_{t-1} + U_g x_t)$$
            - $$c_t = (f * c_{t-1}) + (\mathrm{\tilde{C}}_{t} * i)$$
            - $$h_t = \tanh(c_t) * o$$
        - Forget gate (f) controls how much to keep or forget by outputting 0 to 1
        - Input controls how much of input is used in C (long term memory)
        - H is called short term memory because it is calculated and passed at each step
        - C is modified slowly over time
    3. #### GRU
        - GRU only has reset and update gates
4. ### Encoder Decoder
    - #### Architecture:
    - #### Shortcomings:
        1. 🔴 Sequential Execution: Happens across time step with each next step dependent on previous (slows max
           possible rate of training)
        2. 🔴 Bottle Neck at Encoded Vector: Since context vector is fixed length no matter length of input sequence
        3. 🔴 No long range dependencies: Hidden state decays almost instantly, even long term memory decays across
           sequence
5. ### Attention Mechanism
    - #### How it fixes Shortcomings of encoder decoder:
        - Provides a way for each part of decoder to focus on each part of input
        - 🟢 Allows selective focus on what part that decoder finds useful (Gives long range dependencies)
        - 🟢 No bottlenecks as there is no common context vector
        - 🟢 Allows parallel execution of each segment of decoder (while training)
    - #### Types:
        - Bahdanau Attention
        - Scaled Dot Product
            - $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V $
            - Here Q, K and V are
            - $Q = X W^Q, \quad K = X W^K, \quad V = X W^V$
            - d<sub>k</sub> controls the size of attention matrices i.e Q and K are n * dk and dk * m
            - Root dk scales dot product
            - Softmax-ed to calculate how much should V contribute
            - Training: Trains using teacher forcing mostly

6. ### Transformer
    - ![img_18.png](img_18.png)
    - Usually Made up of encoder decoder (or one of these)
    - Improvement over traditional RNN
        - 🟢 Trained in parallel since each token can look at all others instead of relying on last output (with teacher
          forcing)
        - 🟢 No bottleneck: Since no central encoded vector, each token fetches its context from attending to all others
        - 🟢 Can freely have long range dependencies. Each token can attend to all others.
    - Made up of
        1. **Embedding** [(explained in DataPreProcessing.md)](./DataPreProcessing.md#Word-Embedding)
        2. **Positional** Encoding [(explained in LLM.md)](./LLM.md#positional-encoding)
        3. **MultiHead** Attention layer
        4. **Residual Connections**: Each layer's output is added to it's input (through skip connections) (prevents
           vanishing gradient)
        5. **Layer Normalization**: In Transformers, two major strategies exist for applying LayerNorm:
            - ![img_19.png](img_19.png)
            1. **Post-Normalization** (Post-LN): LayerNorm is applied after the residual connection.
            2. **Pre-Normalization** (Pre-LN): LayerNorm is applied inside the residual connection, before each
               sub-layer.
        6. Feedforward Neural Network: After attention block, each to
7. ### Generative Adversarial Network
8. ### Auto Encoder
9. ### Variational Auto Encoder
10. ### Diffusion Networks