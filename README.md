# Explainable Fake News Detection and Personalised Credible Recommendation via GraphML

The proliferation of fake news on critical topics remains a persistent problem on the web. False information is often used as a means to spread hysteria and panic among readers who are gullible and seldom fact-check. Manual verification itself is a cumbersome task requiring extensive research and critical evaluation, making it time-consuming and impractical for everyday readers.

The objective is to develop a system that allows users to input the URL of a news article and receive a credibility assessment. The algorithm will (i) classify the article as ’real’ or ’fake’, (ii) provide an explanation for the decision, and (iii) recommend three relevant and safe alternatives, selected based on source reliability, content similarity, and user network popularity. This approach will make fact-checking time-efficient, transparent, and accessible.

*This is a course project for CSE 847 Machine Learning at Michigan State University. The project is ongoing.

## Table of Contents
1. [Dataset](#dataset)
2. [Methodology](#methodology)

## Dataset
The [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) dataset contains articles from PolitiFact and GossipCop. Both are fact-checking websites, where PolitiFact focuses on politics, government claims, and public policy, and GossipCop focuses on entertainment industry rumours. The dataset contains the following features:

`id`: unique identifier for each news

`url`: URL of the article from web that published that news 

`title`: title of the news article

`tweet_ids`: Tweet ids (list separated by tab) of tweets sharing the news

## Methodology
The baseline model will employ TF-IDF scores with logistic regression on text data for binary classification. To scale this up, XGBoost will be trained on article embeddings or TF-IDF scores. Additionally, LightGCN will be trained on user-article interactions for graph-based recommendations. The novel approach will experiment with DistilBERT for the binary classification task and concatenate BERT embeddings with LightGCN or GraphSAGE embeddings to create a hybrid fake news detection and article recommendation system. For explainability, SHAP will be applied to text models, and GNNExplainer to graph-based recommendations.
