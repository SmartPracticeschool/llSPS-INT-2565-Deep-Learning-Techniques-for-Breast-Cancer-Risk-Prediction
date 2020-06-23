import os

import numpy as np
from keras.preprocessing import image 

from keras.models import load_model


import tensorflow as tf

global graph
graph=tf.get_default_graph()

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

model = load_model("breastcancer2.h5")

print('Model loaded. Check http://127.0.0.1:5000/')



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        print("current path")

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print("current path",basepath)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        print("upload folder is",file_path)
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
            
            print("prediction ", preds)
        index = ["breast cancer", "no breast cancer"]
        text = "prediction : " + str(index[preds[0]])
        
    return text

    

if __name__ == '__main__':
    app.run(debug=True,threaded = False)