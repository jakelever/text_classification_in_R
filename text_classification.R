
# This code is based on the classification chapter in
# Supervised Machine Learning for Text Analysis in R
# by Emil Hvitfeldt and Julia Silge
# https://smltar.com/mlclassification.html

# Load various libraries for processing the data
library(tidyverse)
library(tidymodels)
library(textrecipes)
library(stopwords)

# Load various libraries to create classifiers
library(discrim)
library(naivebayes)
library(glmnet)

# Load up the dataset (with two columns: label and text)
dataset <- read_csv("asthma_trials_dataset.csv")

# Convert the label column to a factor
dataset$label <- factor(dataset$label)

# Split the data into training and test set
set.seed(42)
data_split <- initial_split(dataset, strata = label)
data_train <- training(data_split)
data_test <- testing(data_split)

######################################
# Part 1: Which words look important #
######################################

# Create a preprocessing pipeline that tokenizes the text
# strips stopwords, filters uncommon words and converts
# the text into binary (also known as one-hot) vectors that
# represent if each word is present in the text
one_hot_preprocessor <- recipe(label ~ text, data = data_train) %>%
  step_tokenize(text) %>%
  step_stopwords(text) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  #step_tfidf(text)
  step_tf(text, weight_scheme="binary")

# Create a logistic regression classifier model
classifier <- logistic_reg(penalty = 0) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

# Fit the logistic regression classifier to the
# training data, using the preprocessing workflow
# to prepare the data
fitted_train <- workflow() %>%
  add_recipe(one_hot_preprocessor) %>%
  add_model(classifier) %>%
  fit(data = data_train)

# Get the coefficients from the logistic regression so that
# we can inspect which words/tokens were most important
coefs <- predict(fitted_train$fit$fit$fit, s=0, type="coefficients")
coefs <- as.data.frame(coefs[,'s1'])
colnames(coefs) <- c('val')
rownames(coefs) <- gsub('tf_text_','',rownames(coefs))

# Pull out the most important tokens (with the highest and lowest
# coefficients in the classifer)
top_neg_tokens <- coefs[order(coefs$val)[1:20],,drop=FALSE]
top_pos_tokens <- coefs[order(coefs$val,decreasing=T)[1:20],,drop=FALSE]

# Print them so we can see the tokens that are most useful
# to the classifier
print(top_neg_tokens)
print(top_pos_tokens)

#############################################
# Part 2: Tuning and evaluating a classifer #
#############################################

# Prepare to split the training set into folds for doing 
# 10-fold cross validation
set.seed(42)
folds <- vfold_cv(data_train)

# Use an alternative preprocessing approach that uses
# term-frequency - inverse-document-frequency (TF-IDF) which
# weights words based on their frequency and how often they
# appear in other documents, and is generally a good representation
tfidf_preprocessor <- recipe(label ~ text, data = data_train) %>%
  step_tokenize(text) %>%
  step_stopwords(text) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text)

preprocess_and_classify_workflow <- workflow() %>%
  add_recipe(tfidf_preprocessor) %>%
  add_model(classifier)

fitted_resamples <- fit_resamples(
  preprocess_and_classify_workflow,
  folds,
  control = control_resamples(save_pred = TRUE)
)

metrics <- collect_metrics(fitted_resamples)
predictions <- collect_predictions(fitted_resamples)

predictions %>%
  group_by(id) %>%
  roc_curve(truth = label, .pred_0) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for US Consumer Finance Complaints",
    subtitle = "Each resample fold is shown in a different color"
  )

conf_mat_resampled(fitted_resamples, tidy = FALSE) %>%
  autoplot(type = "heatmap")















complaints_rec_v2 <- recipe(label ~ text, data = data_train) %>%
  step_tokenize(text) %>%
  step_tokenfilter(text,
                   max_tokens = tune(), min_times = 100) %>%
  step_tfidf(text)

tune_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

sparse_wf_2 <- workflow() %>%
  add_recipe(complaints_rec_v2) %>%
  add_model(tune_spec)

final_grid <- grid_regular(
  penalty(range = c(-4, 0)),
  max_tokens(range = c(1e3, 3e3)),
  levels = c(penalty = 20, max_tokens = 3)
)

set.seed(2020)
tune_rs <- tune_grid(
  sparse_wf_2,
  folds,
  grid = final_grid,
  metrics = metric_set(accuracy, sensitivity, specificity)
)

autoplot(tune_rs) +
  labs(
    color = "Number of tokens",
    title = "Model performance across regularization penalties and tokens",
    subtitle = paste("We can choose a simpler model with higher regularization")
  )

choose_acc <- tune_rs %>%
  select_by_pct_loss(metric = "accuracy", -penalty)

final_wf <- finalize_workflow(sparse_wf_2, choose_acc)

final_fitted <- last_fit(final_wf, data_split)

collect_metrics(final_fitted)

collect_predictions(final_fitted) %>%
  conf_mat(truth = label, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

collect_predictions(final_fitted)  %>%
  roc_curve(truth = label, .pred_1) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for US Consumer Finance Complaints",
    subtitle = "With final tuned lasso regularized classifier on the test set"
  )


library(vip)

complaints_imp <- extract_fit_parsnip(final_fitted$.workflow[[1]]) %>%
  vi(lambda = choose_acc$penalty)

complaints_imp %>%
  mutate(
    Sign = case_when(Sign == "POS" ~ "Less about credit reporting",
                     Sign == "NEG" ~ "More about credit reporting"),
    Importance = abs(Importance),
    Variable = str_remove_all(Variable, "tfidf_consumer_complaint_narrative_"),
    Variable = str_remove_all(Variable, "textfeature_narrative_copy_")
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
    title = "Variable importance for predicting the topic of a CFPB complaint",
    subtitle = paste0("These features are the most important in predicting\n",
                      "whether a complaint is about credit or not")
  )
