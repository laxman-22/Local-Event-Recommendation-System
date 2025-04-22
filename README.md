# Local-Event-Recommendation-System
This repository contains the code used to develop a recommendation system for activities to do in Durham, NC

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
