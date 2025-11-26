# Formulas
1. ## Gradient Descent  
   $\theta_{t+1} \;=\; \theta_t \;-\; \eta \,\nabla_\theta J(\theta_t)$
   1. ### Momentum Based
      $$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta_{t-1})$$
      $$\theta_t = \theta_{t-1} - v_t$$
   1. ### Nesterov Accelerated Gradient Descent
      $$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta_{t-1} - \gamma v_{t-1})$$
2. Activation Functions
   1. Sigmoid
      $$f(x) = \frac{1}{1 + e^{-x}}$$  
   2. Tanh
      $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
   3. ReLU
      $$f(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$
3. 