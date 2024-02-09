# routes.py
from flask_rest import app
from flask_restful import Resource, Api
from ..models.inference import make_inference


api = Api(app)

class DropPrediction(Resource):
    def post(self):
        # Assuming the input data is a JSON object containing the input data for inference
        input_data = [0, 2, 0, 1, 7, 2, 11, 2, 1, 11, 0, 7]
        predictions = make_inference(input_data)
        return predictions


api.add_resource(DropPrediction, '/')
