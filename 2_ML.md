# Machine Learning Topics

1. Fundamentals
    1. **Supervised Learning**
    2. **Unsupervised Learning**
    3. **Semisupervised Learning**
    4. **Reinforcement Learning**
    5. **Selfsupervised Learning**
2. Concepts
    1. **Training**
    2. **Testing**
    3. **Bias Variance Trade-Off**<br>
       The Bias-Variance Trade-Off is a fundamental concept in machine learning that explains the balance between a
       model's ability to make accurate predictions on training data and its ability to generalize to unseen data.
        1. **Bias**: error due to overly simplistic assumptions in the learning algorithm
        2. **Variance**: model's sensitivity to fluctuations in the training data
           The goal is to find a balance between bias and variance<br><br>
    4. **Cross Validation**
       <br> Used to assess the performance and generalization ability of a model.<br>
       i.e It is a model evaluation process to compare models/hyperparameters  
       **K Folds:**
       <br>
        1. Split the dataset into K equal-sized folds (subsets).
        2. For each fold:
            1. Use that fold as the test set.
            2. Use the remaining K-1 folds as the training set.
            3. Train the model on the training set and evaluate on the test set.
        3. Repeat the process K times, each time with a different fold as the test set.
        4. Average the performance metrics (e.g., accuracy, RMSE) across all folds for a final score.
           <br>Pros & Cons:
           <br>✅ Provides a better estimate of model performance.
           <br>✅ Helps detect overfitting or underfitting.
           <br>⚠️ Computationally expensive for large datasets.
    5. **Loss Function**
    6. **Evaluation Metrics**
        <br>TP = True Positives
        <br>TN = True Negatives
        <br>FP = False Positives
        <br>FN = False Negatives
        1. Accuracy
           The percentage of total correct predictions.
           $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
            <br>✅ Simple, intuitive.
           ⚠️ Can be misleading with imbalanced datasets.
        2. Precision
            <br>Proportion of correctly predicted positive cases out of all predicted positives.
           $$\text{Precision} = \frac{TP}{TP + FP}$$
        3. Recall
           Proportion of actual positive cases that were correctly predicted.
           $$\text{Recall} = \frac{TP}{TP + FN}$$
        4. F1-Score
            <br>Harmonic mean of Precision and Recall. Best used when there's class imbalance.
            $$\text{F1Score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
            $$\text{F1Score} = \frac{2TP}{2TP + FP + FN}$$
        5. ROC-AUC
        <br>How well the model distinguishes between the two classes across all thresholds.
            1. ROC Curve
           <br> Plots True Positive Rate vs False Positive Rate
           TPR = TP/TP+FN   FPR = FP/FP+TN
           2. AUC: Area Under the ROC Curve. Higher is better
           <br>There’s no single formula for ROC-AUC, as it’s calculated using numerical integration
        