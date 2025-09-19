# Transformer

- ![img_18.png](img_18.png)
- Usually Made up of encoder decoder (or one of these)
- Improvement over traditional RNN
    - 🟢 Trained in parallel since each token can look at all others instead of relying on last output (with teacher
      forcing)
    - 🟢 No bottleneck: Since no central encoded vector, each token fetches its context from attending to all others
    - 🟢 Can freely have long range dependencies. Each token can attend to all others.

## Layers

## 1. **Embedding** [(explained in DataPreProcessing.md)](./DataPreProcessing.md#Word-Embedding)

## 2. **Positional** Encoding [(explained in LLM.md)](./LLM.md#positional-encoding)

## 3. **MultiHead** Attention layer

## 4. **Residual Connections**:

Each layer's output is added to it's input (through skip connections) (prevents vanishing gradient)

## 5. **Layer Normalization**

In Transformers, two major strategies exist for applying LayerNorm

- ![img_19.png](img_19.png)

1. ### **Post-Normalization** (Post-LN):
    - LayerNorm is applied after the residual connection.
    - This strategy was used in orignal papers
    - ![img_20.png](img_20.png)
        - i.e Input $ x $ goes through sub-layer (e.g., attention $ A(x) $ or feed-forward $ F(x) $).
        - Residual: $ x + \text{sub-layer}(x) $.
        - Then apply LayerNorm: $ \text{LN}(x + \text{sub-layer}(x)) $.
        - This is repeated across $ L $ layers.
    - 🔴 This can affect the signal from skip connection since it's normalized (across N encoders distorts greatly)
    - 🔴 Causes gradient instability: Why? At init outputs of sublayer (attention with linear) are small (due to He init)
      so residual path
      dominates forcing both to same range and multiply over deepstacks.(Thus attention output dilutes)
    - 🔴 Requires learning rate warm up
2. **Pre-Normalization** (Pre-LN):
    - LayerNorm is applied inside the residual connection, before each sub-layer.
        - i.e. Apply LayerNorm first: $ \text{LN}(x) $
        - Sub-layer on normalized input: $ \text{sub-layer}(\text{LN}(x)) $.
        - Residual: $ x + \text{sub-layer}(\text{LN}(x)) $.
        - No additional norm after the residual in the basic pre-norm variant.
    - 🟢 Does not distort residual connection
    - 🟢 No need for learning rate warmup
    -
    - Used in most modern models (GPT, BERT)
3. Hybrid: Both pre and post do exist (called Sandwich Normalization)

## 6. Position Wise Feedforward Neural Network:

After attention block, usually a two layer fully connected neural network applied independently to each token
embedding.  
FFN(x)=W2​f(W1​x+b1​)+b2​

- W1 usually expands to a much larger representation (4x in BERT) before w2 squeezes it back down
- 🟢 Transform the output of attention, basically processing it (basically now i have all the needed info, what should i
  do with it)
- 🟢 Add non-linearity to the output of attention blocks
 ## 7. Softmax
At the end of decoder to transform into probabilities