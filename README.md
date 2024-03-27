# IMDb-Reviews-Sentiment-Classification


IMDb-Reviews-Sentiment-Classification
Introduction
This project develops a text classification model to determine the sentiment (positive or negative) of IMDb movie reviews. Using Logistic Regression and Term Frequency-Inverse Document Frequency (TF-IDF) for feature extraction, the model aims to accurately classify the sentiment of textual movie reviews.

Model Overview
The model is trained using Logistic Regression, a powerful linear classification algorithm suitable for binary classification tasks. For feature extraction, the TF-IDF technique is employed to convert text data into a format that can be effectively used by the machine learning model.

Dataset
The model uses the IMDb reviews dataset, a large collection of movie reviews from the IMDb website, widely used for binary sentiment classification tasks. This dataset is publicly available through the Hugging Face datasets library, making it easily accessible for research and educational purposes.

The dataset consists of 50,000 reviews split evenly into 25,000 training and 25,000 testing samples. Each review has been labeled as either positive or negative, providing a balanced dataset for training and evaluating the sentiment analysis model. The reviews vary in length and cover a broad range of movies, offering a diverse set of textual data for model training.

Usage
To utilize the trained sentiment analysis model for classifying new IMDb reviews, follow these simple steps:

Load the Trained Model: Initially, load the trained Logistic Regression model. This step typically involves loading the model from a serialized file format, such as a .pkl (Pickle) file, which stores the trained model parameters.

Prepare the Review Text for Classification: Before making predictions, it's essential to preprocess the new review text to match the format expected by the model. This preprocessing step includes tokenizing the text and transforming it using the same TF-IDF vectorizer applied during the training phase. Ensure that the text is cleaned and tokenized in the same manner as the training data to maintain consistency.

Classify the Sentiment of the Review: With the model loaded and the review text appropriately preprocessed, you can now classify the sentiment of the review. The model will output a sentiment prediction, indicating whether the review is positive or negative.

Performance
The performance of the sentiment analysis model is evaluated based on several key metrics, including accuracy, precision, recall, and F1-score. These metrics provide a comprehensive overview of the model's ability to accurately classify IMDb reviews as positive or negative.

Accuracy: Measures the proportion of correctly classified reviews out of the total number of reviews. Precision: Indicates the accuracy of positive predictions, measuring the proportion of true positive predictions in all positive predictions. Recall: Assesses the model's ability to identify all relevant instances, calculating the proportion of true positive predictions in all actual positives. F1-Score: Provides a balance between precision and recall, representing the harmonic mean of the two metrics. These performance metrics reflect the model's effectiveness in sentiment classification tasks and its generalizability to unseen movie reviews.

- The model achieved an accuracy of 88%, a precision of 89% for the positive class, a recall of 87%, and an F1-score of 88%.
