# inference.py

import joblib
import pandas as pd

# function to return actual path of the model from base path
def get_model_path(model_file):
    model_path = 'flask_rest/models/' + model_file
    return model_path


def load_models():
    """
    Load all the saved models.
    Returns a dictionary containing model names as keys and the corresponding loaded model as values.
    """
    models = {}
    model_files = [
        "random_forest_model.joblib",
        "logistic_regression_model.joblib",
        "k-nearest_neighbors_model.joblib",
        "decision_tree_model.joblib",
        "gaussian_naive_bayes_model.joblib",
        "multilayer_perceptron_model.joblib",
        "support_vector_machine_model.joblib",
        "linear_discriminant_analysis_model.joblib",
        "adaboost_model.joblib"
    ]

    for model_file in model_files:
        model_name = model_file.split('_model')[0].title().replace('_', ' ')
        model = joblib.load( get_model_path(model_file) )
        models[model_name] = model

    return models

def make_inference(input_data, models):
    """
    Make predictions using all loaded models.
    :param input_data: Pandas DataFrame containing input data for inference.
    :return: Dictionary containing model names as keys and corresponding predictions as values.
    """

    predictions = {}

    for model_name, model in models.items():
        # 'input_data' is a DataFrame with the same columns as the training data
        model_predictions = model.predict([input_data])
        predictions[model_name] = str(model_predictions[0])

    print(predictions)

    return predictions

