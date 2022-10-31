# Predict Churn Rate for Sparkify, a Digital Musical Servives

Sparkify is a popular digital musical services where users can listen to their favourite songs either using the free tier or purchase premium subscription model in which
they stream music as free but pay a monthly flat rate. Understanding the behaviour and preference of users helping us in business planning. The activities carried out by the users such as logging in and out, downgrade, upgrade or services cancellation are recorded as data which can be used to predict the risk to churn for users.

Exploratory Analysis

The relationship between gender and level of users - (paid or free) against churned users are explored.

Feature Engineering

Positive and negative feedbacks, interaction between users, and trends of song and artists are investigated to identify the tendency of churn rate. The target and feature variables are defined and prepared for the use in modeling.

Modeling

Predicting churn is a classification in machine learning. 3 classification models - Logistic Regression, Random Forest and Gradient Boosting Tree are chosen to compare the model performance and validate the suitable hyperparameters. For evalaution metrics, accuracy and F1 score are evaluated. Accuracy measures how well the model in predicitng churn and F1 score measures the accuracy of the model's performance.
Spark SQL, Spark DataFrame and Spark ML will be used in this analysis.
