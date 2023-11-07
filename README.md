# Football Player Valuation Prediction
## Problem and Project Description
In the realm of professional football, a player transfer takes place when a player, under contractual obligations, transitions from one club to another. This intricate process involves the official relocation of a player's registration from their current football club to a new one. Typically, the transfer initiation occurs when a representative from an interested club officially inquires with the club where their prospective player is currently registered. If the selling club expresses an openness to the idea, negotiations commence for a transfer fee. These negotiations are often facilitated by intermediaries and involve determining the financial compensation to be paid by the acquiring club. However, price negotiations between clubs are time-consuming and lack standardization. Nowadays, the player's current club often aims to maximize the price for their player, contributing to significant inflation in the football player market value.

Our project is dedicated to the prediction of football players' market values, drawing insights from their in-game, profile, and attribute statistics. Utilizing machine learning techniques, we aim to provide a valuable tool for clubs, agents, and enthusiasts, enabling them to assess and comprehend the market value of players. The predictive player valuation model can help professional clubs to setting a reasonable starting point for negotiations regarding a player's price.

## Dataset
The dataset of this project taken from [Kaggle Dataset link](https://www.kaggle.com/datasets/davidcariboo/player-scores). In this project we dont use all of available data from the kaggle. The used dataset: apperances.csv, games.csv, players.csv. The used dataset can check and download in this repository [Used Dataset](./data-raw)

## Dependencies 
To run this project, you will need the following dependencies:
* Python 3.9
* Flask==3.0.0
* gunicorn==21.2.0
* scikit-learn==1.3.0

Project dependencies can be installed by running:
```python
pip install -r requirements.txt
```

Or alternatively can create virtual environtment from prepared pipfile using **Pipenv** :
1. Create enviroment from pipfile : 
```python
pipenv install
```

3. Enter the created environment :
```python
pipenv shell
```

## Model Creation

## Model Deployment on Web Service:
The trained model should be deploy for ease access of the model functionality. One of the most effective method is we serve the model as web service. In this project we use **Flask** (an python web service  framework). 

To run the web service locally:

1. Use gunicorn to serve the webserice in production scheme




## Acknowledgement
This repo is intended to submission of midterm project of mlzoomcamp.
