# Campus Dropout Prediction Documentation

## Overview

Welcome to the Campus Dropout Prediction, a  machine learning project designed to predict the likelihood of students dropping out based on various input features. This documentation aims to guide users through the installation process, usage details, and understanding of the model's output.

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
  "location_name": 1,
  "home_language": 2,
  "hh_occupation": 0,
  "hh_edu": 1,
  "hh_size": 7,
  "school_distanceKm": 2,
  "age": 11,
  "gender": 2,
  "mothers_edu": 1,
  "grade": 11,
  "meansToSchool": 0,
  "hh_children": 7
}
```

## Output Format

The predictions are returned in JSON format with model names as keys and corresponding predictions as values.

Example:

```json
{
  "predictions": {
    "Random Forest": "1",
    "Logistic Regression": "1",
    "K-Nearest Neighbors": "1",
    "Decision Tree": "1",
    "Gaussian Naive Bayes": "1",
    "Multilayer Perceptron": "1",
    "Support Vector Machine": "1",
    "Linear Discriminant Analysis": "1",
    "Adaboost": "1"
  }
}
```

## Models

The project utilizes several sklearn machine learning models for making predictions, including Random Forest, Logistic Regression, K-Nearest Neighbors, Decision Tree, Gaussian Naive Bayes, Multilayer Perceptron, Support Vector Machine, Linear Discriminant Analysis, and Adaboost.

## Error Handling

Inference is done via POST requests to the `/` endpoint. If the input data is not in the correct format, the server will return a 400 Bad Request error with a JSON response containing the error message.

## Contributors

- [Eddie Gulay](https://eddiegulay.github.io/)