# routes.py
from flask_rest import app, load_models, make_inference
from flask_restful import Resource, Api, reqparse



models = load_models()


api = Api(app)
class DropPrediction(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        # Add argument parsing for each input feature
        self.parser.add_argument('location_name', type=int)
        self.parser.add_argument('home_language', type=int)
        self.parser.add_argument('hh_occupation', type=int)
        self.parser.add_argument('hh_edu', type=int)
        self.parser.add_argument('hh_size', type=int)
        self.parser.add_argument('school_distanceKm', type=int)
        self.parser.add_argument('age', type=int)
        self.parser.add_argument('gender', type=int)
        self.parser.add_argument('mothers_edu', type=int)
        self.parser.add_argument('grade', type=int)
        self.parser.add_argument('meansToSchool', type=int)
        self.parser.add_argument('hh_children', type=int)

    def post(self):
        # Parse the JSON data from the POST request
        data = self.parser.parse_args()

        # Convert the parsed data into a list for inference
        input_data = [
            data['location_name'],
            data['home_language'],
            data['hh_occupation'],
            data['hh_edu'],
            data['hh_size'],
            data['school_distanceKm'],
            data['age'],
            data['gender'],
            data['mothers_edu'],
            data['grade'],
            data['meansToSchool'],
            data['hh_children']
        ]

        # Make inference using the models
        predictions = make_inference(input_data, models=models)

        return {'predictions': predictions}

api.add_resource(DropPrediction, '/')
