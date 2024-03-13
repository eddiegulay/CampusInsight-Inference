# inference.py

import joblib
import pandas as pd

""" Global """
dropout_mapping = {0: 'No', 1: 'Yes'}


# function to return actual path of the model from base path
def get_model_path(model_file):
    model_path = f'flask_rest/models/{model_file}'
    return model_path

def get_encoder_path(encoder_file):
    encoder_path = f'flask_rest/encoders/{encoder_file}'
    return encoder_path

def load_models():
    """
    Load all the saved models.
    Returns a dictionary containing model names as keys and the corresponding loaded model as values.
    """
    models = {}
    model_files = [
        "AdaBoost.joblib",
        "BaggingClassifier.joblib",
        "DecisionTree.joblib",
        "GaussianNaiveBayes.joblib",
        "GradientBoostingClassifier.joblib",
        "KNeighborsClassifier.joblib",
        "LinearDiscriminant Analysis.joblib",
        "LogisticRegression.joblib",
        "MultilayerPerceptron.joblib",
        "RandomForest.joblib"
    ]

    for model_file in model_files:
        model_name = model_file
        model = joblib.load( get_model_path(model_file) )
        models[model_name] = model

    return models

def make_inference(input_data, models):
    """
    Make predictions using all loaded models.
    :param input_data: Pandas DataFrame containing input data for inference.
    :return: Dictionary containing model names as keys and corresponding predictions as values.
    """
    encoder = joblib.load(get_encoder_path('general_encoder.joblib'))

    predictions = {}

    for model_name, model in models.items():
        # 'input_data' is a DataFrame with the same columns as the training data
        predicted_class, predicted_probabilities = make_prediction(target_data, model, encoder, dropout_mapping)
        predictions[model_name] = {
            'predicted_class': predicted_class,
            'predicted_probabilities': list(predicted_probabilities)
        }

    print(predictions)

    return predictions


def make_prediction(target_data, clf, encoder, dropout_mapping):
    # Convert the dictionary to a DataFrame
    target_data_for_inference = pd.DataFrame(target_data)
    transform_data = target_data_for_inference.apply(decode_row, axis=1)
    one_hot_transform_data = encoder.transform(target_data_for_inference.drop('age', axis=1))
    inference_one_hot_transform_data = np.append(one_hot_transform_data[0], transform_data['age'][0])
    inference_data = np.array(inference_one_hot_transform_data)

    try:
        predictions = clf.predict([inference_data.astype(str)])
        probabilities = clf.predict_proba([inference_data.astype(str)])
    except:
        predictions = clf.predict([inference_data])
        probabilities = clf.predict_proba([inference_data])

    return dropout_mapping[predictions[0]], probabilities[0]
