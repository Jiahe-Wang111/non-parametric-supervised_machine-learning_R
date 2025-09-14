# non-parametric-supervised_machine-learning_R：Predictive Modeling with KNN, Decision Trees, and Random Forests
This project demonstrates the application of three machine learning algorithms—K-Nearest Neighbors (KNN), Decision Trees, and Random Forests—to two distinct classification tasks: predicting social media ad clicks and identifying tweet authors.

The focus is on non-parametric models, exploring how they handle prediction tasks where relationships may be non-linear or high-dimensional. Compared with parametric approaches (e.g., logistic regression, GAMs), these methods make fewer distributional assumptions and directly learn patterns from the data.

## Project Structure

### Part 1: Social Network Ad Purchase Prediction

**Objective**: Predict whether a user will purchase a product based on Age, Gender, and Estimated Salary.

**Models Used**:
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

**Key Steps**:
- Data preprocessing: Factor encoding, train-test splitting, and standardization (for KNN)
- Hyperparameter tuning via cross-validation (`k` for KNN, `cp` for Decision Trees, `mtry` for Random Forest)
- Evaluation of accuracy and model complexity trade-offs
- Visualization of decision boundaries and variable importance

**Takeaway**:
- KNN performs well in low-dimensional spaces but deteriorates with higher dimensions.
- Decision Trees offer interpretable rules but are prone to overfitting.
- Random Forests improve predictive performance and stability through ensemble learning, while also providing key predictor insights.

---

### Part 2: Predicting Tweet Authors (Bernie Sanders vs. Donald Trump)

**Objective**: Classify tweet authorship based on text content.

**Models Used**:
- KNN
- Decision Tree
- Random Forest

**Key Steps**:
- Constructing a document-term matrix (DTM) from tweet text
- Training models on high-dimensional text features
- Cross-validation for model evaluation
- Random Forest variable importance analysis to identify influential words

**Takeaway**:
- KNN struggles with high-dimensional sparse data due to the curse of dimensionality.
- Decision Trees capture non-linear patterns but are unstable.
- Random Forests offer the best generalization, improved accuracy, and interpretable word importance.

---

## Key Learning Points

- **KNN**: A simple non-parametric classifier that works well in low dimensions but doesn’t scale well to high-dimensional data.
- **Decision Trees**: Highly interpretable and capable of modeling complex non-linear relationships, but prone to overfitting.
- **Random Forests**: An ensemble method that improves stability and accuracy, with built-in variable importance measures.
- Cross-validation is essential for hyperparameter tuning and fair model comparison.
- Non-parametric models are powerful tools for capturing complex and non-linear patterns in data.

---

## Results Highlights

Both projects highlight the strengths and limitations of each algorithm across different data types (structured vs. text). Random Forests consistently provided the best performance and interpretability, especially in high-dimensional settings.

---

Developed for educational purposes in machine learning and predictive modeling.

