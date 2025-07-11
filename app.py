# importing libraries
import numpy as np
import pandas as pd
import random
import io
import flask
import pickle
import base64
from scipy import ndimage
from PIL import Image
from sklearn_som.som import SOM
from flask import Flask, render_template, request

# create instance of the class
app = Flask(__name__)


def rename_filetype(before):
    if before == 'image/jpeg':
        after = "JPEG"
    elif before == 'image/png':
        after = "PNG"
    else:
        err = 'Invalid file type'
        return render_template('index.html', error_message=err)
    return after


def center_object(img):
    # find center of mass
    y_center, x_center = ndimage.center_of_mass(img)
    y_center = int(y_center)
    x_center = int(x_center)

    # determine distances from center of mass to edge of image
    y_dist_to_edge = min(y_center, len(img) - y_center)
    x_dist_to_edge = min(x_center, len(img[0]) - x_center)
    dist_to_edge = max(y_dist_to_edge, x_dist_to_edge)

    # zero padding
    if x_dist_to_edge > y_dist_to_edge:
        # if overhang is on top
        if y_center - x_dist_to_edge < 0:
            overhang = x_dist_to_edge - y_center
            # zero pad on top
            img = np.append([[0] * len(img[0])] * overhang, img, axis=0)
            # move y center after padding on top
            y_center += overhang
        # else overhang is on bottom
        else:
            overhang = x_dist_to_edge - y_dist_to_edge
            # zero pad on bottom
            img = np.append(img, [[0] * len(img[0])] * overhang, axis=0)
    if y_dist_to_edge > x_dist_to_edge:
        # if overhang is on left
        if x_center - y_dist_to_edge < 0:
            overhang = y_dist_to_edge - x_center
            # zero pad on left
            img = np.append([[0] * overhang] * len(img), img, axis=1)
            # move x center after padding on left
            x_center += overhang
        # else overhang is on right
        else:
            overhang = y_dist_to_edge - x_dist_to_edge
            # zero pad on right
            img = np.append(img, [[0] * overhang] * len(img), axis=1)

    # crop image as square
    img = img[y_center - dist_to_edge:y_center + dist_to_edge, x_center - dist_to_edge:x_center + dist_to_edge]
    return img.astype(np.uint8)


def preprocess(img):
    # edge detection (subtract Gaussian blurred image from original)
    img = img - ndimage.gaussian_filter(img, 3)
    # center object
    img = center_object(img)
    # convert back to Image object
    img = Image.fromarray(img)

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
def html_prep(img, file_type):
    img = Image.fromarray(img.astype("uint8"))
    raw_bytes = io.BytesIO()
    img.save(raw_bytes, file_type)
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
        # define allowed file types
        allowed_file_types = ['image/jpeg', 'image/png']
        if file.content_type not in allowed_file_types:
            err = 'Invalid file type'
            return render_template('index.html', error_message=err)
        # input validation
        if file.filename == '':
            err = 'No file selected'
            return render_template('index.html', error_message=err)

        if file:
            # retrieve submitted image
            image_to_map = Image.open(file.stream)
            # rename file type
            filetype = rename_filetype(file.content_type)
            # ensure jpeg images are RGB only
            if filetype == "JPEG":
                image_to_map = image_to_map.convert('RGB')
            # prepare submitted image for results view
            submitted = html_prep(np.array(image_to_map), filetype)
            # convert image to grayscale
            image_to_map = image_to_map.convert('L')
            # preprocess submitted image
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
                next_image = np.reshape(related_images[n], (28, 28))
                related[i] = html_prep(next_image, filetype)

            return render_template('result.html', img_sub=submitted,
                                    rel=related)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)