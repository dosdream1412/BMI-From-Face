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


app = Flask(__name__)
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app, resources={r"/*": {"origins": "*"}})
# net = load_model('model-resnet50-final.h5')



def detect_faces(image):
    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()
    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]
    return face_frames

@app.route('/')
def hello_world2():
    testJ = {"thin": 'hello', "dd": 'world'}
    # strJson = str(testJ)
    Res = json.dumps(testJ)
    return (Res)
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
            #face.show()
        imgload = tt.load_img(fp, target_size=(224, 224))
        #imgload.show()
        cls_list = ['อ้วน', 'น้ำหนักเกิน', 'น้ำหนักปกติ', 'ผอมเกินไป', 'อ้วนมาก']  # edit
        #cls_list = ['fat', 'littleFat', 'normal', 'thin', 'veryFat']
        # load the trained model
        net = load_model('model-resnet50-final.h5')
        x = tt.img_to_array(imgload)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        pred = net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        arrJson = []
        for i in top_inds:
            cal = "{0:.3f}".format(pred[i])
            fileNameMain = open(os.getcwd()+'\\'+cls_list[i]+'\\main.txt',encoding="utf8")
            fileNameFood = open(os.getcwd()+'\\'+cls_list[i]+'\\food.txt',encoding="utf8")
            fileNameExercise = open(os.getcwd()+'\\'+cls_list[i]+'\\exercise.txt',encoding="utf8")
            j = {'per': cal, 'class': cls_list[i],'main':fileNameMain.read(),'food':fileNameFood.read(),'exercise':fileNameExercise.read()}
            arrJson.append(j)
        #imgload.show()
        #strJson = str(arrJson[0])
        Res = json.dumps(arrJson[0])
        os.remove(fp)
        return (Res)



if __name__ == '__main__':
    
    app.run(host='0.0.0.0',port=5000,threaded=False)
