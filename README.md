# Football Player Valuation Prediction
## Problem and Project Description
In the realm of professional football, a player transfer takes place when a player, under contractual obligations, transitions from one club to another. This intricate process involves the official relocation of a player's registration from their current football club to a new one. Typically, the transfer initiation occurs when a representative from an interested club officially inquires with the club where their prospective player is currently registered. If the selling club expresses an openness to the idea, negotiations commence for a transfer fee. These negotiations are often facilitated by intermediaries and involve determining the financial compensation to be paid by the acquiring club. However, price negotiations between clubs are time-consuming and lack standardization. Nowadays, the player's current club often aims to maximize the price for their player, contributing to significant inflation in the football player market value.

This project is dedicated to the prediction of football players' market values, drawing insights from their in-game, profile, and attribute statistics. Utilizing machine learning techniques, we aim to provide a valuable tool for clubs, agents, and enthusiasts, enabling them to assess and comprehend the market value of players. The predictive player valuation model can help professional clubs to setting a reasonable starting point for negotiations regarding a player's price.

## Dataset
The dataset of this project taken from [Kaggle Dataset link](https://www.kaggle.com/datasets/davidcariboo/player-scores). In this project we dont use all of available data from the kaggle. The used dataset: apperances.csv, games.csv, players.csv. The used dataset can check and download in this repository , spesifically on [data-raw](./data-raw) folder.

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
The selected model in this project is:
* Linear Regression

  Linear regression is a fundamental statistical technique used for modeling the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. It aims to find the best-fitting line (or hyperplane in multi-dimensional space) that minimizes the sum of squared differences between predicted and actual values, making it a valuable tool for making predictions and understanding the linear relationships within data.

* Decision Tree

  A decision tree is a supervised machine learning algorithm used for both classification and regression tasks. It works by recursively partitioning the dataset into subsets based on the most significant attribute at each level, resulting in a tree-like structure. Decision trees are interpretable and can make predictions by traversing the tree from the root to a leaf node, providing insights into decision-making processes in a visual and intuitive way.

* XGBoost

  XGBoost (Extreme Gradient Boosting) is a powerful and efficient ensemble machine learning algorithm that combines the strengths of gradient boosting and tree-based methods. It's widely used for both classification and regression tasks and is known for its speed, scalability, and effectiveness in improving predictive accuracy. XGBoost builds an ensemble of decision trees, continually improving the model's performance by minimizing the loss function and handling overfitting through regularization techniques.

### Evaluation Metrics
In this project, we use RMSE as the metric
he Root Mean Squared Error (RMSE) is a measure of the average deviation between predicted values and actual values in a dataset. It is often used to evaluate the accuracy of a predictive model.

The formula for RMSE is:


$$\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$


Where:
- $n$ is the total number of data points.
- $y_i$ represents the actual (observed) value for data point \(i\).
- $\hat{y}_i$ represents the predicted value for data point \(i\).

The RMSE value quantifies the typical error or "residuals" between actual and predicted values, with lower RMSE values indicating a better-fitting model.

### Used Features
* height_in_cm: Player's height.
* goals_2022: Goals scored by the player in the 2022 season.
* games_2022: Games played by the player in the 2022 season.
* assists_2022: Assists created by the player in the 2022 season.
* minutes_played_2022: Total minutes of play by the player in the 2022 season.
* goals_for_2022: Total goals scored by the player's club in the 2022 season.
* goals_against_2022: Total goals conceded by the player's club in the 2022 season.
* clean_sheet_2022: Total clean sheets kept by the player's club in the 2022 season.
* age: Player's age.

### Best Model
From the exploration, we determined that **XGBoost** is the best model, achieving RMSE scores of **0.859** for the validation data and **0.891** for the test data. For detailed exploration, please refer to [this notebook](./notebook.ipynb)

## Model Deployment on Web Service:
The trained model should be deploy for ease access of the model functionality. One of the most effective method is we serve the model as web service. In this project we use **Flask** (an python web service  framework). The implementation of the flask web service in this repo can found [here](./predict.py)

To run the web service locally:

1. Go to root directory of the project

2. Serve using gunicorn with below command:
```python
gunicorn --bind 0.0.0.0:9696 predict:ap
```
3. Hit the prepared predict api route with post method and send the required payload input :
<img width="480" alt="image" src="https://github.com/ryanpram/player-valuation-prediction/assets/34083758/93d23c4b-d1f8-40ee-a8d4-78a82d178e92">
<img width="860" alt="image" src="https://github.com/ryanpram/player-valuation-prediction/assets/34083758/b3835d4f-8018-467a-8161-b04f7b6b4bff">


## Model Deployment on Cloud (AWS):
### Docker Containerization:
Before we deploy our model web service , we wrap it with docker container first. Docker container can wrap all dependencies needed with the apps so we can avoid dependencies conflic when we run our app in the cloud environment.
1. Install Docker desktop 
2. Build docker image from [Dockerfile script](./predict.py)
```python
docker build -t <tag-name> <Dockerfile-location-path>
```
3. Make sure the docker image already created successfully with:
```python
docker images
```
4. Run docker images on port 9696
```python
docker run -p 9696:9696 <tag-name>
```
When docker container is running , we can access the flask web service inside the container through 9696 port.

### Cloud Deployment:
After we successfully wrap our web service app in docker container, we are ready now to deploy to cloud environtment. Here we use one of cloud computing service in AWS, which is Elastic Beanstalk. Elastic Beanstalk is a Platform-as-a-Service (PaaS) offering by AWS that simplifies application deployment and management. It allows us to easily deploy, monitor, and scale web applications and services without dealing with the underlying infrastructure, making it a convenient choice for quickly launching web applications.

Step by step AWS Elastic Benstalk (EBS) deployment:
1. Init EBS application
```python
 eb init -p "Docker running on 64bit Amazon Linux 2" -r <region-code-name> <desired-application-name>
```
2. Test eb server running well locally
```python
 eb local run --port 9696
```
3. Create environtment and deploy our apps on it
```python
 eb create <desired-env-name>
```

This project already deployed to AWS EBS that can be accessed on:
```python
 [POST] http://player-valuation2-env.eba-hi5iceym.ap-southeast-1.elasticbeanstalk.com/predict
```
<img width="1071" alt="image" src="https://github.com/ryanpram/player-valuation-prediction/assets/34083758/bc5adc7b-fe24-4d78-a34a-e23092d8124b">

Example of the way to access our deployed model on cloud
<img width="806" alt="image" src="https://github.com/ryanpram/player-valuation-prediction/assets/34083758/668b02ab-e55c-42a8-9dc1-d46da6623b7c">
<img width="489" alt="image" src="https://github.com/ryanpram/player-valuation-prediction/assets/34083758/4a9e68c2-352a-4c25-947f-7c281a053307">


## Acknowledgement
This repo is intended to submission of midterm project of mlzoomcamp.
