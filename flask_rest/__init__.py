# __init__.py
from flask import Flask
from flask_cors import CORS
from .inference import load_models, make_inference

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

from flask_rest import routes
