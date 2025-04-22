from flask import Flask, render_template, jsonify
import cv2
from imutils.video import VideoStream
import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)


# Load face detector and mask detector model
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector_model.keras")

# Initialize video stream
vs = VideoStream(src=0).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_mask')
def detect_mask():
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Detect faces and predict masks
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    results = []

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        result = {
            "label": label,
            "color": color,
            "box": (startX, startY, endX, endY)
        }
        results.append(result)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
