# importing libraries
import numpy as np
import pandas as pd
import random
import io
import flask
import pickle
import base64
from PIL import Image
from sklearn_som.som import SOM
from flask import Flask, render_template, request

# create instance of the class
app = Flask(__name__)


def preprocess(img):
    # make image square
    w, h = img.size
    if w > h:
        h = w
    else:
        w = h
    img = img.resize((w, h))
    # scale image
    img.thumbnail((28, 28))
    img = np.array(img)
    # flatten image array
    img = np.reshape(img, 784)
    # double array to meet prediction requirements
    img = np.concatenate(([img], [img]))
    return img


# prediction function
def get_prediction(img):
    # prepare image for mapping (first column is the label)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    prediction = loaded_model.predict(img)
    return prediction[0]


# prepare image to send to html file
def html_prep(img):
    img = Image.fromarray(img.astype("uint8"))
    raw_bytes = io.BytesIO()
    img.save(raw_bytes, "JPEG")
    encoded_image = "data:image/png;base64," + base64.b64encode(raw_bytes.getvalue()).decode('ascii')
    return encoded_image


# to tell flask what url should trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # input validation
        if 'image' not in request.files:
            err = 'No file attached in request'
            return render_template('index.html', error_message=err)
        # retrieve file submitted
        file = request.files['image']
        # input validation
        if file.filename == '':
            err = 'No file selected'
            return render_template('index.html', error_message=err)

        if file:
            # retrieve submitted image
            image_to_map = Image.open(file.stream).convert('L')
            # prepare submitted image for results view
            submitted = html_prep(np.array(image_to_map))
            # preprocess submitted
            image_to_map = preprocess(image_to_map)

            # make prediction from submitted image
            node = get_prediction(image_to_map)

            # retrieve images and their assigned nodes
            train = pd.read_csv("train.csv")
            train = np.array(train)
            train_nodes = pd.read_csv("fashion_prediction.csv")
            train_nodes = np.array(np.reshape(train_nodes, len(train_nodes)))

            # find images related to submitted image
            related_images = train[train_nodes == node]

            # retrieved related images and prepare for results view
            related = [None]*15
            sample_index = random.sample(range(len(related_images)), min(15, len(related_images)))
            for i, n in enumerate(sample_index):
                related[i] = html_prep(np.reshape(related_images[n], (28, 28)))

            return render_template('result.html', img_sub=submitted,
                                    rel=related)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)