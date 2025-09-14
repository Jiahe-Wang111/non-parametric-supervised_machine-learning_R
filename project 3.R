################################################################################

library(data.table)
library(caret)
library(ggplot2)
library(rpart.plot)
library(dplyr)
library(randomForest)

################################################################################

# PART 1: SOCIAL NETWORK AD PURCHASE PREDICTION
## load dataset (example dataset provided in ./data/)
social_data <- fread("data/Kaggle_Social_Network_Ads.csv")
social_data$Purchased <- as.factor(social_data$Purchased)
social_data <- social_data[, -c("user_id")]

## K-Nearest Neighbors (KNN)
tc <- trainControl(method = "cv", number = 10)

set.seed(123)
knn_model <- train(Purchased ~ .,
                   data = social_data,
                   method = "knn",
                   tuneGrid = expand.grid(k = c(1,2,3,5,10,20,25,50)),
                   preProcess = c("center", "scale"),
                   trControl = tc)
knn_model

ggplot(knn_model$results, aes(x = k, y = Accuracy)) +
  geom_line() +
  labs(title = "KNN: Accuracy vs k",
       x = "Number of Neighbors (k)",
       y = "Accuracy")


## Decision Tree
set.seed(123)
rpart_model <- train(Purchased ~ .,
                     data = social_data,
                     method = "rpart",
                     tuneGrid = expand.grid(cp = seq(0, 0.1, 0.01)),
                     trControl = tc,
                     control = rpart.control(minbucket = 10))
rpart_model

ggplot(rpart_model$results, aes(x = cp, y = Accuracy)) +
  geom_line() +
  labs(title = "Decision Tree: Accuracy vs cp",
       x = "Complexity Parameter (cp)",
       y = "Accuracy")

best_tree <- rpart_model$finalModel
rpart.plot(best_tree)

varimp_tree <- varImp(rpart_model)  # Variable importance
plot(varimp_tree, main = "Variable Importance - Decision Tree")


## Random Forest
set.seed(123)
rf_model <- train(Purchased ~ .,
                  data = social_data,
                  method = "rf",
                  tuneGrid = expand.grid(mtry = c(1,2,3)),
                  ntree = 200,
                  trControl = tc)
rf_model

ggplot(rf_model$results, aes(x = mtry, y = Accuracy)) +
  geom_line() +
  labs(title = "Random Forest: Accuracy vs mtry",
       x = "Number of Variables Tried at Each Split (mtry)",
       y = "Accuracy")

varimp_rf <- varImp(rf_model)
plot(varimp_rf, main = "Variable Importance - Random Forest")

################################################################################

# Part 2: Predicting Tweet Authors (Bernie Sanders vs. Donald Trump)
## Load dataset
trump_data <- fread("data/trumpbernie2.csv")
trump_data$trump_tweet <- as.factor(trump_data$trump_tweet)
tc_text <- trainControl(method = "cv", number = 5)

## Decision Tree
set.seed(123)
trump_tree <- train(trump_tweet ~ .,
                    data = trump_data,
                    method = "rpart",
                    tuneGrid = expand.grid(cp = c(0, 0.001, 0.01, 0.1)),
                    trControl = tc_text)
trump_tree

ggplot(trump_tree$results, aes(x = cp, y = Accuracy)) +
  geom_line() +
  labs(title = "Decision Tree: Accuracy vs cp",
       x = "Complexity Parameter (cp)",
       y = "Accuracy")    # Accuracy vs cp

rpart.plot(trump_tree$finalModel)
varimp_tree_trump <- varImp(trump_tree)
plot(varimp_tree_trump, top = 20, main = "Top 20 Important Words (Decision Tree)")

# Random Forest
set.seed(123)
mtry_values <- c(5, 10, 25, 50, 100)

trump_rf <- train(trump_tweet ~ .,
                  data = trump_data,
                  method = "rf",
                  tuneGrid = expand.grid(mtry = mtry_values),
                  ntree = 200,
                  trControl = tc_text)
trump_rf

ggplot(trump_rf$results, aes(x = mtry, y = Accuracy)) +
  geom_line() +
  labs(title = "Random Forest: Accuracy vs mtry",
       x = "Number of Variables per Split (mtry)",
       y = "Accuracy")   # Accuracy vs mtry

varimp_rf_trump <- varImp(trump_rf)
plot(varimp_rf_trump, top = 20, main = "Top 20 Important Words (Random Forest)")

# Extract top 20 from each model
top_dt <- varimp_tree_trump$importance %>%
  as.data.frame() %>%
  tibble::rownames_to_column("var") %>%
  arrange(desc(Overall)) %>%
  slice(1:20) %>%
  mutate(model = "Decision Tree")

top_rf <- varimp_rf_trump$importance %>%
  as.data.frame() %>%
  tibble::rownames_to_column("var") %>%
  arrange(desc(Overall)) %>%
  slice(1:20) %>%
  mutate(model = "Random Forest")

combined <- bind_rows(top_dt, top_rf)

ggplot(combined, aes(x = Overall, y = reorder(var, Overall), color = model)) +
  geom_point(size = 3) +
  facet_wrap(~model, scales = "free_y") +
  labs(title = "Top 20 Important Words: Decision Tree vs Random Forest",
       x = "Importance",
       y = "Words")

################################################################################

# Conclusion

## KNN performs well for small values of k, but accuracy decreases as k grows.
## Decision Trees are interpretable but may overfit if cp is too small.
## Random Forests generally achieve higher accuracy and more stable results, while also providing robust variable importance measures.