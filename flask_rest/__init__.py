# __init__.py
from flask import Flask
from .inference import load_models, make_inference

app = Flask(__name__)

from flask_rest import routes
