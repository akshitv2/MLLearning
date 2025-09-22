# ML Questions

0. **how do decision trees handle missing values**
    - Surrogate Splits: When the best splitting feature has missing values, the tree looks for another feature that produces a similar split (a surrogate).
    - For categorical variables, the algorithm can treat "missing" as its own category.
    - XGBoost picks the direction that gives the best gain in the objective function (e.g., log loss, MSE) during training.
      - This “default direction” is stored in the tree.
1. **What is the bias-variance tradeoff in machine learning, and why is it important?**
   - The bias-variance tradeoff describes the balance between two sources of error in a machine learning model:
     - Bias is the error from overly simplistic assumptions in the model. A high-bias model underfits the data, failing to capture important patterns. 
     - Variance is the error from too much sensitivity to the training data. A high-variance model overfits, capturing noise and performing poorly on unseen data. 
   - The tradeoff is important because minimizing total error requires finding the right balance:
   - Too much bias → poor accuracy on both training and test data.
   - Too much variance → great training accuracy but poor generalization.
   - In practice, techniques like regularization, cross-validation, ensemble methods, and careful model selection are used to manage this tradeoff and achieve good generalization.
2. **Can you explain the difference between bagging and boosting in ensemble learning?**
    - Both bagging and boosting are ensemble methods that combine multiple weak learners (often decision trees) to improve performance, but they differ in how they build and combine models:
    - 