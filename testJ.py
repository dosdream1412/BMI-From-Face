from flask import Flask, render_template, request
import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing import image as tt
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import json
from flask_cors import CORS

from Predict import detect_faces

app = Flask(__name__)
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def hello_world2():
    return ("Fuck")
    # return render_template('testBMI.html')

@app.route('/upload-pic')
def upload_file():
    return render_template('testBMI.html')

@app.route('/forecast', methods=['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['pic']
        fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(fp)
        bmi = io.imread(fp)
        # Detect faces
        detected_faces = detect_faces(bmi)
        for n, face_rect in enumerate(detected_faces):
            face = Image.fromarray(bmi).crop(face_rect)
            plt.subplot(1, len(detected_faces), n + 1)
            plt.axis('off')
            face.save(fp)
            # face.show()
        imgload = tt.load_img(fp, target_size=(224, 224))
        imgload.show()
        testJ = {"thin" : 'hello2' , "dd" : 'world'}
        #strJson = str(testJ)
        Res = json.dumps(testJ)
        return (Res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)