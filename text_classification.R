
# This code is based on the classification chapter in
# Supervised Machine Learning for Text Analysis in R
# by Emil Hvitfeldt and Julia Silge
# https://smltar.com/mlclassification.html

# Load various libraries for preprocessing, building and interrogating models
library(tidyverse)
library(tidymodels)
library(textrecipes)
library(stopwords)
library(glmnet)
library(vip)

# Load up the dataset (with two columns: label and text)
dataset <- read_csv("asthma_trials_dataset.csv")

# Set the factor levels. The first level is considered the
# positive class (used for calculating metrics later)
dataset$label <- factor(dataset$label, levels=c('pos','neg'))

# Split the data into training and test set
set.seed(42)
data_split <- initial_split(dataset, strata = label)
data_train <- training(data_split)
data_test <- testing(data_split)

# Use a preprocessing approach that splits the text into tokens
# removes common words (stop words) and creates count data of words
# using term-frequency - inverse-document-frequency (TF-IDF) which
# weights words based on their frequency and how often they
# appear in other documents, and is generally a good representation
tfidf_preprocessor <- recipe(label ~ text, data = data_train) %>%
  step_tokenize(text) %>%
  step_stopwords(text) %>%
  step_tfidf(text)
  
# Set up a logistic regression classifier where the amount of
# regularisation (a control on the complexity of the model) is
# a parameter that will be tuned
classifier <- logistic_reg(penalty = tune(), mixture=1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

# Create a workflow by combing the tokenising/tfidf step with the classifier
preprocess_and_classify_workflow <- workflow() %>%
  add_recipe(tfidf_preprocessor) %>%
  add_model(classifier)

# Set out how we will optimise the penalty parameter
final_parameter_grid <- grid_regular(
  penalty(range = c(-4, 0)),
  levels = c(penalty = 20)
)

# Prepare to split the training set into folds for doing 
# 10-fold cross validation
set.seed(42)
folds <- vfold_cv(data_train)

# Run through different values of the parameters using the 10-fold
# cross validation to see what penalty gives the best performance
parameter_search_results <- tune_grid(
  preprocess_and_classify_workflow,
  folds,
  grid = final_parameter_grid,
  metrics = metric_set(accuracy, sensitivity, specificity, f_meas)
)

# Plot out the effect of regularization on different performance metrics
autoplot(parameter_search_results) +
  labs(
    color = "Number of tokens",
    title = "Model performance across regularization penalties",
    subtitle = paste("We can choose a simpler model with higher regularization")
  )

# Choose the parameter set that gives the best F1 score
parameter_choice <- parameter_search_results %>%
  select_best(metric = "f_meas")

# Create a workflow with the optimal choice of penalty
final_workflow <- finalize_workflow(preprocess_and_classify_workflow, parameter_choice)

# Fit the dataset on the training data and evaluate on the test data
final_fitted <- last_fit(final_workflow, data_split, metrics=metric_set(roc_auc, accuracy, sensitivity, specificity, f_meas))

# Print out the performance metrics
collect_metrics(final_fitted)

# Output a confusion matrix of predictions on the dataset
collect_predictions(final_fitted) %>%
  conf_mat(truth = label, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

# Plot an ROC curve
collect_predictions(final_fitted)  %>%
  roc_curve(truth = label, .pred_pos) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve",
    subtitle = "With final tuned lasso regularized classifier on the test set"
  )

# Extract the importances of the different features (the tokenized words)
# from the final fitted model
token_importances <- extract_fit_parsnip(final_fitted$.workflow[[1]]) %>%
  vi(lambda = parameter_choice$penalty)

# Plot the most important words used by the model to predict the class
token_importances %>%
  filter(Importance != 0) %>%
  mutate(
    Sign = case_when(Sign == "POS" ~ "Importance for predicting negative class",
                     Sign == "NEG" ~ "Important for predicting positive class"),
    Importance = abs(Importance),
    Variable = str_remove_all(Variable, "tfidf_text_"),
  ) %>%
  group_by(Sign) %>%
  top_n(20, Importance) %>%
  ungroup %>%
  ggplot(aes(x = Importance,
             y = fct_reorder(Variable, Importance),
             fill = Sign)) +
  geom_col(show.legend = FALSE) +
  scale_x_continuous(expand = c(0, 0)) +
  facet_wrap(~Sign, scales = "free") +
  labs(
    y = NULL,
    title = "Importance of input tokens to predicting positive/negative class",
  )
