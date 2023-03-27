# Example pipeline of a binary text classification approach in R

This repo contains an example script of text classification based on the [Classification chapter](https://smltar.com/mlclassification.html) of [Supervised Machine Learning for Text Analysis in R by Emil Hvitfeldt and Julia Silge](https://smltar.com). 

## Dataset

The example dataset is a small set of asthma-related publications from [PubMed](https://pubmed.ncbi.nlm.nih.gov/) and a label of whether it describes a clinical trial (based of [MeSH data](https://www.nlm.nih.gov/mesh/meshhome.html)). The file ([asthma_trials_dataset.csv](https://github.com/jakelever/text_classification_in_R/blob/main/asthma_trials_dataset.csv)) is a comma-delimited file with two columns: label and text.

## Script

The [text_classification.R](https://github.com/jakelever/text_classification_in_R/blob/main/text_classification.R) mostly follows the final steps in the [Classification chapter](https://smltar.com/mlclassification.html). It does the following:

- It tokenizes the text, removes [stop words](https://en.wikipedia.org/wiki/Stop_word) and normalizes the word counts using [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to take into account word frequency and rarity.
- It splits the data into training and test sets
- It creates a logistic regression classifier and tunes the [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) on the training set (using k-fold cross validation)
- It then uses the optimal hyperparameters to fit the final model and reports the performance through various metrics, a confusion matrix and an ROC curve.
- Finally, it uses the [vip package](https://koalaverse.github.io/vip/articles/vip.html) to get the importance of the input tokens to the predictions.
