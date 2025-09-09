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
               TPR = TP/TP+FN FPR = FP/FP+TN
            2. AUC: Area Under the ROC Curve. Higher is better
               <br>There’s no single formula for ROC-AUC, as it’s calculated using numerical integration
3. Data Preprocessing & Feature Engineering
    1. Data Cleaning
    2. Handling Missing Data
    3. Encoding Categorical Variables
    4. Feature Scaling (Normalization, Standardization)
    5. Feature Selection & Dimensionality Reduction
    6. PCA (Principal Component Analysis)
    7. t-SNE, UMAP
    8. Outlier Detection
    9. Data Augmentation (especially in images)
4. Supervised Learning
    1. Regression
       1. Linear Regression<br>
       Linear regression analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable.
        It helps predict outcomes by fitting a straight line to observed data points, making it easy to interpret and apply.
       <br>To calculate best-fit line linear regression uses a traditional slope-intercept form which is given below
       <br>y=w<sup>⊤</sup>x+b
       <br>
       In regression, the difference between the observed value of the dependent variable(y i ) and the predicted value(predicted) is called the residuals.
       Assumptions: 
       2. Polynomial Regression 
       3. Ridge, Lasso, Elastic Net 
       4. Decision Trees for Regression  
       A decision tree is a flowchart-like tree structure.  
       * Each internal node represents a test on a feature.
       * Each branch represents an outcome of the test.
       * Each leaf node represents a class label (in classification) or a value (in regression)
       5. Support Vector Regression (SVR)
       6. Ensemble Methods (Bagging, Boosting for regression)
       7. Gaussian Processes
        