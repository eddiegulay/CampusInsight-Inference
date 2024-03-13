# routes.py
from flask_rest import app, load_models, make_inference
from flask_restful import Resource, Api, reqparse



models = load_models()


api = Api(app)
class DropPrediction(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        # Add argument parsing for each input feature
        self.parser.add_argument('location_name', type=str)
        self.parser.add_argument('home_language', type=str)
        self.parser.add_argument('hh_occupation', type=str)
        self.parser.add_argument('hh_edu', type=str)
        self.parser.add_argument('hh_size', type=str)
        self.parser.add_argument('school_distanceKm', type=str)
        self.parser.add_argument('age', type=int)
        self.parser.add_argument('gender', type=str)
        self.parser.add_argument('mothers_edu', type=str)
        self.parser.add_argument('grade', type=str)
        self.parser.add_argument('meansToSchool', type=str)
        self.parser.add_argument('hh_children', type=str)

    def post(self):
        # Parse the JSON data from the POST request
        data = self.parser.parse_args()

        target_data_for_inference = {
            'location_name': [data['location_name']],
            'home_language': [data['home_language']],
            'hh_occupation': [data['hh_occupation']],
            'hh_edu': [data['hh_edu']],
            'hh_size': [data['hh_size']],
            'school_distanceKm': [data['school_distanceKm']],
            'age': [data['age']],
            'gender': [data['gender']],
            'mothers_edu': [data['mothers_edu']],
            'grade': [data['grade']],
            'meansToSchool': [data['meansToSchool']],
            'hh_children': [data['hh_children']]
        }


        # Make inference using the models
        predictions = make_inference(target_data_for_inference, models=models)

        return {'predictions': predictions}

api.add_resource(DropPrediction, '/')
