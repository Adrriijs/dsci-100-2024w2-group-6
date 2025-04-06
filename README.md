# Customer Subscription Prediction Using K-Nearest Neighbors

This repository contains our group project for **DSCI 100** at the University of British Columbia. In this project, we use the K-Nearest Neighbors (KNN) classification algorithm to predict whether a customer will subscribe to a product based on their profile data.

## 📁 Project Overview

The goal of this project is to explore and apply a supervised machine learning technique on a real-world-style dataset. We chose KNN for its simplicity, interpretability, and effectiveness in handling classification tasks with small to medium-sized datasets.

We walk through the full data science process using the `tidymodels` framework in R, including:

- Data cleaning and preprocessing  
- Train-test data splitting  
- Feature engineering using recipes  
- KNN model tuning using cross-validation  
- Model evaluation with accuracy and confusion matrix  
- Visualizing prediction results

## 📊 Dataset

The dataset used in this project contains customer demographic and marketing interaction data. The target variable is whether the customer subscribed to the product (`subscription`), which is a binary classification problem.

**Note:** This dataset was provided for educational use in the DSCI 100 course and may not reflect a real commercial dataset.

## 🧪 Methods

- **Train/Test Split:** We split the data 80/20 to preserve enough data for training while holding out unseen examples for evaluation.
- **Preprocessing:** Categorical features were one-hot encoded using `step_dummy()`, and numeric features were scaled using `step_normalize()`, to ensure fair distance calculations for KNN.
- **Modeling:** We used `nearest_neighbor()` from the `parsnip` package with the `kknn` engine.
- **Tuning:** Hyperparameter tuning was done using 5-fold cross-validation to find the best number of neighbors (`k`).
- **Evaluation:** Final model performance was evaluated on the test set using accuracy and a confusion matrix.
