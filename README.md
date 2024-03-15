# Campus Dropout Prediction Documentation

## Overview

Welcome to the Campus Dropout Prediction, a  machine learning project designed to predict the likelihood of students dropping out based on various input features. This documentation aims to guide users through the installation process, usage details, and understanding of the model's output.

![image](https://github.com/eddiegulay/CampusInsight-Inference/assets/88213379/9e75e1e6-8755-43f4-b040-2ce205021dd3)


## Table of Contents

- [Campus Dropout Prediction Documentation](#campus-dropout-prediction-documentation)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
    - [Making Predictions](#making-predictions)
    - [API Endpoints](#api-endpoints)
  - [Input Data Format](#input-data-format)
  - [Output Format](#output-format)
  - [Models](#models)
  - [Error Handling](#error-handling)
  - [Contributors](#contributors)

## Installation

### Prerequisites

- Python 3.6 or higher
- Flask
- Flask-RESTful
- Scikit-learn
- Joblib

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### Setup

1. Clone the project repository:

```bash
git clone https://github.com/eddiegulay/CampusInsight-Inference.git
cd CampusDropoutPrediction
```

2. Run the Flask application:

```bash
python run.py
```

The application will be accessible at `http://localhost:5000/`.

## Usage

### Making Predictions

To make predictions, send a POST request to the `/` endpoint with JSON data in the [specified format](#input-data-format).

Example using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"location_name": 1, "home_language": 2, ...}' http://localhost:5000/
```

### API Endpoints

- **POST `/`**: Make dropout predictions based on input data.

## Input Data Format

The input data should be a JSON object with the following format:

```json
{
    "location_name": "Urban",
    "home_language": "English",
    "hh_occupation": "Private sector",
    "hh_edu": "None",
    "hh_size": "More than five",
    "school_distanceKm": "7-10 km",
    "age": 16,
    "gender": "Female",
    "mothers_edu": "None",
    "grade": "Form One",
    "meansToSchool": "Walk",
    "hh_children": "More than five"
}

```

## Output Format

The predictions are returned in JSON format with model names as keys and corresponding predictions as values.

Example:

```json
{
  "predictions": {
    "AdaBoost": {
      "predicted_class": "Yes",
      "predicted_probabilities": [
        0.4820458943856379,
        0.5179541056143622
      ]
    },
    "BaggingClassifier": {
      "predicted_class": "Yes",
      "predicted_probabilities": [
        0.0,
        1.0
      ]
    },
    "DecisionTree": {
      "predicted_class": "Yes",
      "predicted_probabilities": [
        0.0,
        1.0
      ]
    },
    "GaussianNaiveBayes": {
      "predicted_class": "Yes",
      "predicted_probabilities": [
        1.525064456842568e-221,
        1.0
      ]
    },
    "GradientBoostingClassifier": {
      "predicted_class": "No",
      "predicted_probabilities": [
        0.7427914618026694,
        0.25720853819733064
      ]
    },
    "KNeighborsClassifier": {
      "predicted_class": "No",
      "predicted_probabilities": [
        0.6,
        0.4
      ]
    },
    "LinearDiscriminantAnalysis": {
      "predicted_class": "Yes",
      "predicted_probabilities": [
        0.002812547334127169,
        0.9971874526658728
      ]
    },
    "LogisticRegression": {
      "predicted_class": "Yes",
      "predicted_probabilities": [
        0.07167075943804024,
        0.9283292405619598
      ]
    },
    "MultilayerPerceptron": {
      "predicted_class": "Yes",
      "predicted_probabilities": [
        7.95127821529018e-06,
        0.9999920487217847
      ]
    },
    "RandomForest": {
      "predicted_class": "Yes",
      "predicted_probabilities": [
        0.42,
        0.58
      ]
    }
  }
}
```

## Models

The project utilizes several sklearn machine learning models for making predictions, including Random Forest, Logistic Regression, K-Nearest Neighbors, Decision Tree, Gaussian Naive Bayes, Multilayer Perceptron, Support Vector Machine, Linear Discriminant Analysis, and Adaboost.

## Error Handling

Inference is done via POST requests to the `/` endpoint. If the input data is not in the correct format, the server will return a 400 Bad Request error with a JSON response containing the error message.

## Contributors

- [Eddie Gulay](https://eddiegulay.github.io/)
