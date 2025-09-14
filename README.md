# non-parametric-supervised_machine-learning_R
The focus is on non-parametric models, exploring how they handle prediction tasks where relationships may be non-linear or high-dimensional. Compared with parametric approaches (e.g., logistic regression, GAMs), these methods make fewer distributional assumptions and directly learn patterns from the data.

Project Structure
Part 1: Social Network Ad Purchase Prediction

Objective: Predict whether a user will purchase a product based on Age, Gender, and Salary.

Models: K-Nearest Neighbors (KNN), Decision Tree, Random Forest

Key Steps:

Data preprocessing (factor encoding, train/test split, standardization for KNN)

Hyperparameter tuning with cross-validation (k for KNN, cp for decision trees, mtry for random forest)

Evaluation of model accuracy and complexity trade-offs

Visualization of decision boundaries and variable importance

Takeaway:

KNN performs adequately on low-dimensional data but deteriorates with more features.

Decision trees provide interpretable splitting rules but are prone to overfitting.

Random forests improve predictive performance and stability through ensembling, while also highlighting key predictors.

Part 2: Predicting Tweet Authors (Bernie Sanders vs. Donald Trump)

Objective: Predict the author of a tweet (Trump or Bernie) based on word features.

Models: KNN, Decision Tree, Random Forest

Key Steps:

Constructing a document-term matrix (DTM) from tweet text

Training each model on high-dimensional text features

Model evaluation using cross-validation

Random forest variable importance analysis to identify influential words

Takeaway:

KNN struggles in high-dimensional sparse data (curse of dimensionality).

Decision trees can pick up non-linear word patterns but remain unstable.

Random forests generalize best, offering improved accuracy and interpretable word-level importance.

Key Learning Points

KNN: Simple baseline for non-parametric classification, but limited scalability in high dimensions.

Decision Trees: Intuitive and interpretable, suitable for modeling non-linear relationships, but sensitive to overfitting.

Random Forests: Robust ensemble method that balances bias-variance tradeoff and provides stable variable importance measures.

Cross-validation is essential for hyperparameter tuning and model comparison.

Non-parametric models complement parametric approaches by capturing complex, non-linear structures in the data.
