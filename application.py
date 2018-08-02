import numpy as np
import clfs
from im_utils import normalize, get_box, square, resize, gray2rgb
from PIL import Image
import flask
import io
import base64
import tensorflow as tf
# from flask_login import login_required, current_user
from flask_cors import CORS
# import logging

# Define global parameters
model = None
PA_model = ['densenet121', 'linear']
PA_weights_path = 'weights/PA.hdf5'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])



# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
CORS(app)

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    PA_clf = clfs.get_model(model=PA_model[0], top=PA_model[1])
    PA_clf.load_weights(PA_weights_path)
    model = PA_clf
    global graph
    graph = tf.get_default_graph()


def preprocess(im, flip=False, new_size=224):
    im = im[:,:,0]
    im = normalize(im, flip=flip)
    im = normalize(get_box(im))
    im = square(im)
    im = resize(im, new_x=new_size, new_y=new_size)  # using default new_x=224,new_y=224
    im = gray2rgb(im)  # change to 3-channel mode
    return im


app.add_url_rule('/', 'index', (lambda: 'Hello'))


@app.route("/predict", methods=["POST"])
def predict():
    message = flask.request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    image = np.array(image)
    processed_image = preprocess(image)

    global graph
    with graph.as_default():
        prediction = model.predict(np.expand_dims(processed_image, axis=0)).tolist()

    response = {
        'prediction': {
            "abnormality": prediction[0][1]
        }
    }

    return flask.jsonify(response)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    # app.debug = True
    app.run()