# app.py

from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import io
from boundingBoxDrawer import boundingBoxDrawer
from ensembleModel import ensembleModel
from objDetectionRCNN import objDetectionRCNN
from objDetectionY8 import objDetectionY8

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        #Read the image-----------------------------------------------
        npimg = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        #Get a copy of the image--------------------------------------
        image = img.copy()

        #Perform object detection with YoloV8-------------------------
        detection_list_Y8 = objDetectionY8(image) 

        #Perform object detection with FRCNN--------------------------
        detection_list_RCNN = objDetectionRCNN(image)

        #Ensemble two models------------------------------------------
        detection_list = ensembleModel(detection_list_Y8, detection_list_RCNN)
        print(detection_list)

        #Draw bounding boxes------------------------------------------
        objDetImg = boundingBoxDrawer(detection_list, image)
        
        #Convert the image back to bytes for display------------------
        _, img_encoded = cv2.imencode('.jpg', image) #objDetImg)
        img_bytes = img_encoded.tobytes()

        return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
