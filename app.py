#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask_bootstrap import Bootstrap

import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template

from models import UploadForm
from predictor import predict_breed
from keras.models import load_model

CURRENT_DIR = os.path.dirname(__file__)


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
bootstrap = Bootstrap(app)
model = load_model(f'saved_models/weights.best.Resnet50.hdf5')


@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.upload.data
        filename = secure_filename(f.filename)
        file_url = os.path.join('static', filename)
        f.save(file_url)
        form = None
        prediction = predict_breed(model, file_url)
    else:
        file_url = None
        prediction = None
    return render_template("index.html", form=form, file_url=file_url, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
