# Local-Event-Recommendation-System
This repository contains the code used to develop a recommendation system for activities to do in Durham, NC. This app is deployed on streamlit cloud and can be found [here](https://laxman-22-local-event-recommendation-system-app-nffwnu.streamlit.app/)

## Problem
It's pretty difficult to find activities to do and sometimes the most popular thing isn't what everyone wants to do. This recommendation system is able to tailor activities based on your preferences and learns the more you use it.

## Data Sources
The activities have been hand picked by me in the auto_events.csv file and theres approximately 120 activities that have been curated, other fields have been filled in using Google Places API and Gemini leveraging the search feature.

Sample user ratings have been synthetically generated to simulate past user behaviour as there wasn't a user base to go off of. All other data is used by the code for other mappings.

## Model Evaluation Process and Metric Selection
The evaluation method selected for this case was NDCG (Normalized Discounted Cumulative Gain) which essentially scores how relevant the ranking of activities are to the user. Below are the results for the evaluation of these models

### Naive Approach
```
Mean Squared Error: 2.01166419821211
Average NDCG: 0.9092436281695736
```
MSE was used here to check how accurate the rating predictions were based on the tags. In this case it seems to do pretty well but NDCG is more relevant.

Based on the recommendations provided by the naive approach a score of 0.90
suggests that the recommendations are highly relevant and in a good ranking in terms of the rating of the activities.

The limitation here is that there is no personalization as every user gets the same recommendations and there's no room for implementing a feedback loop
### KNN Regressor Model
```
Average NDCG: 0.8750988599477756
```
Performance here drops a bit in terms of NDCG, however it is still providing highly relevant items in a good order in terms of rank.

However, this model was used in the end for the front end as it is computationally efficient, accurate enough and it is computationally feasible to implement a feedback loop in real time.

The feedback loop updates the ratings that the user provides and refits the model to that data to pick up on these changes and give more relevant activities based on that.

The model was trained on predicting the ratings from tag-based features and some numerical ones too. This structure was used to build a new user profile and using cosine similarity we can compare the user profile vector with all of the activities and get a list of the most relevant ones.

### Deep Learning Autoencoder
```
Final Average NDCG: 0.8833941054910444
Test Reconstruction Loss: 0.1880
```
Similarly, this autoencoder also had solid performance but it was more computationally intensive to retrain the model with new data to update in near real time so the KNN model was selected.

### Results

Overall, all models had good performance, however it is important to note that this data is synthetic so it definitely has flaws. Furthermore, the dataset of events is not exhaustive so that is certainly an area that would have to be developed further.

However, as a starting point, this model certainly gives accurate recommendations and updates in real time based on live feedback which is super useful.

### Ethics Statement
All of the data collected was from publicly available sources. There is no PII (Personal Identifying Information) and there are certain biases that were identified and acknowledged in this study. This system is intended for educational and research purposes only, and it should not be used to harm anyone or sell anything.

## Getting Started

1) The first step is to clone this repo in order to get access to all necessary files
```(bash)
git clone https://github.com/laxman-22/Local-Event-Recommendation-System.git
```
2) Secondly, create a new virtual environment to isolate dependencies
```(bash)
python -m venv venv
```
3) Next, activate the virtual environment and install requirements
```(bash)
# Linux
source venv/bin/activate

# Windows
./venv/Scripts/activate

pip install -r requirements.txt
```
3) Below are the different scripts that can be ran to test out the models
```(bash)
cd models

# Run the Naive model
python naive_approach.py

# Run the Non Deep Learning (KNN Regressor) model
python non_deep_learning.py

# Run the Deep Learning (Autoencoder) model
python deep_learning.py

```

> [!NOTE]
> The KNN Regressor model creates embeddings which are saved and used to make predictions, however the feedback mechanism requires retraining which is performed to update the model. The deep learning model works in a similar fashion but feedback was not implemented due to computational overhead.

> [!NOTE]
> Another thing to mention is that most of the data besides auto_events.csv is
synthetically generated due to a lack of data and no user base for collection.

> [!IMPORTANT]
> Do not run automated_collection or gemini_data_enrichment python scripts. They were heavily manually edited during the data collection process as it was semi-automated.
