import joblib
import pandas as pd

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
        model = joblib.load(model_file)
        models[model_name] = model

    return models

def make_inference(input_data):
    """
    Make predictions using all loaded models.
    :param input_data: Pandas DataFrame containing input data for inference.
    :return: Dictionary containing model names as keys and corresponding predictions as values.
    """
    models = load_models()
    predictions = {}

    for model_name, model in models.items():
        # Assuming 'input_data' is a DataFrame with the same columns as the training data
        # Adjust the input_data accordingly based on your specific needs
        model_predictions = model.predict(input_data)
        predictions[model_name] = model_predictions[0]

    return predictions

if __name__ == "__main__":
    # Example usage if you run inference.py directly
    # Load your input data for inference (replace 'input_data.csv' with your actual file path)
    input_data = [0, 2, 0, 1, 7, 2, 11, 2, 1, 11, 0, 7]

    # Make predictions using all models
    all_predictions = make_inference([input_data])



print(all_predictions)
