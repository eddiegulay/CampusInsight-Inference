# inference.py

import joblib
import pandas as pd
import numpy as np

""" Global """
dropout_mapping = {0: 'No', 1: 'Yes'}

gender_mapping = {1: 'Male', 2: 'Female'}
home_language_mapping = {0: 'Kiswahili', 1: 'English', 2: 'Native language'}
hh_occupation_mapping = {0: 'Other', 1: 'Unemployed', 2: 'Agriculture', 3: 'Self-employed', 4: 'Public sector', 5: 'Private sector', 6: 'Housewife'}
hh_children_mapping = {0: 'None', 1: 'Two Children', 2: 'Three Children', 3: 'Four Children', 4: 'Five Children', 5: 'More than five'}
mothers_edu_mapping = {0: 'None', 1: 'Primary', 2: 'Secondary', 3: 'Postsecondary'}
hh_edu_mapping = {0: 'None', 1: 'Primary', 2: 'Secondary', 3: 'Postsecondary'}
meansToSchool_mapping = {0: 'Walk', 1: 'Bicycle/motorbike', 2: 'Public transport', 3: 'Private car'}
location_name_mapping = {0: 'Rural', 1: 'Urban'}
grade_mapping = {9: 'Form One', 10: 'Form Two', 11: 'Form Three', 12: 'Form Four'}
dropout_mapping = {0: 'No', 1: 'Yes'}
school_distance_mapping = {1: '0-0.5 km', 2: '0.5-1 km', 3: '1-2 km', 4: '2-3 km', 5: '4-5 km', 6: '6-7 km', 7: '7-10 km', 8: 'More than 11 km'}

# Reverse mappings
gender_mapping_reverse = {v: k for k, v in gender_mapping.items()}
home_language_mapping_reverse = {v: k for k, v in home_language_mapping.items()}
hh_occupation_mapping_reverse = {v: k for k, v in hh_occupation_mapping.items()}
hh_children_mapping_reverse = {v: k for k, v in hh_children_mapping.items()}
mothers_edu_mapping_reverse = {v: k for k, v in mothers_edu_mapping.items()}
hh_edu_mapping_reverse = {v: k for k, v in hh_edu_mapping.items()}
meansToSchool_mapping_reverse = {v: k for k, v in meansToSchool_mapping.items()}
location_name_mapping_reverse = {v: k for k, v in location_name_mapping.items()}
grade_mapping_reverse = {v: k for k, v in grade_mapping.items()}
dropout_mapping_reverse = {v: k for k, v in dropout_mapping.items()}
school_distance_mapping_reverse = {v: k for k, v in school_distance_mapping.items()}


def encode_row(row, dropout=False):
    row = row.copy()
    row['gender'] = gender_mapping[row['gender']]
    row['home_language'] = home_language_mapping[row['home_language']]
    row['hh_occupation'] = hh_occupation_mapping[row['hh_occupation']]
    row['hh_children'] = hh_children_mapping[row['hh_children']]
    row['mothers_edu'] = mothers_edu_mapping[row['mothers_edu']]
    row['hh_edu'] = hh_edu_mapping[row['hh_edu']]
    row['meansToSchool'] = meansToSchool_mapping[row['meansToSchool']]
    row['location_name'] = location_name_mapping[row['location_name']]
    row['grade'] = grade_mapping[row['grade']]
    row['school_distanceKm'] = school_distance_mapping[row['school_distanceKm']]
    row['hh_size'] = hh_children_mapping[row['hh_size']]
    if dropout:
        row['dropout'] = dropout_mapping[row['dropout']]
    return row

def decode_row(row, dropout=False):
    row = row.copy()
    row['gender'] = gender_mapping_reverse[row['gender']]
    row['home_language'] = home_language_mapping_reverse[row['home_language']]
    row['hh_occupation'] = hh_occupation_mapping_reverse[row['hh_occupation']]
    row['hh_children'] = hh_children_mapping_reverse[row['hh_children']]
    row['mothers_edu'] = mothers_edu_mapping_reverse[row['mothers_edu']]
    row['hh_edu'] = hh_edu_mapping_reverse[row['hh_edu']]
    row['meansToSchool'] = meansToSchool_mapping_reverse[row['meansToSchool']]
    row['location_name'] = location_name_mapping_reverse[row['location_name']]
    row['grade'] = grade_mapping_reverse[row['grade']]
    row['school_distanceKm'] = school_distance_mapping_reverse[row['school_distanceKm']]
    row['hh_size'] = hh_children_mapping_reverse[row['hh_size']]
    if dropout:
        row['dropout'] = dropout_mapping[row['dropout']]

    return row

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
        "LinearDiscriminantAnalysis.joblib",
        "LogisticRegression.joblib",
        "MultilayerPerceptron.joblib",
        "RandomForest.joblib"
    ]

    for model_file in model_files:
        model_name = model_file
        model = joblib.load( get_model_path(model_file) )
        models[model_name] = model

    return models

def make_inference(target_data, models):
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
        model_name = model_name.split('.')[0]
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
