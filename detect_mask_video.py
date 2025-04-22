#import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import time
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []


	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence(i.e probability) associated with the detection
		confidence = detections[0, 0, i, 2]

		
                # filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > 0.5:
			
                        # compute the (x,y)- coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			
                        # ensure bounding box fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			
                        # extract the face ROI, convert convert it from BGR to RGB channel ordering, resize it to 224X224, and process it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			
                        # add the face and bounding boxes to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		
                # for faster inference we'll make batch predections on "all" faces at the same time rather than one-by-one predections in the above 'for' loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	
        # return a 2-tuple of face locations and their coressponding locations
	return (locs, preds)
# the above method return the location and predictions. Locations means x and y coordinates of the particular rectangle surrounding the face and
# the predections is the accuracy of the person wearing the mask or not. predictions be like 90% he is wearing the mask or 99% he is wearing.



# load our serialize face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
#firstly we load the faceNet(it is just a parameter). It is used to load the face detector files. Here we use readNet method which is under CV2 in dnn(deep neuro network) module 


# load the face mask detector modek from disk
maskNet = load_model("mask_detector_model.keras")
# load_model is used to load the mask_detector_model from disk

#initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# inside videostream there is a parameter src(source). source is nothing it just our camera if we are using external camera then we can give index1,2,3,4.....
# or when we are using our primary camera then src=0 and start method actually loads the camera.
 


# loop over the frames from the video stream
while True:     # while true we are reading the frames without images
        # grab the frame from the threaded video stream and resize it to have a maximum witdth of 400px

	frame = vs.read()
	frame = imutils.resize(frame, width=400)   # after reading the frame, the frame(dialogue box type thing) will be open named as frame and width=400px

	
        # detect faces in the frame and determine if they are wearing a mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	
        #loop over the detected face locations and their corresponding locations
	for (box, pred) in zip(locs, preds):
		 # unpack the boundary box and predictions
		(startX, startY, endX, endY) = box   # here startX, startY, endX, endY are like x1,y1,x2,y2 coordinates used to make a rectangle
		(mask, withoutMask) = pred

		
                # determine the class label and colour we'll use to draw the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)  # here we use bgr colour pattern

		#include the probability in label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		
                # display the label and bounding box rectangle on the output frame
		cv2.putText(frame, label, (startX, startY - 10),# startX and startY are the coordinates of the text. StartY-10 is done s that text donot overlap to the box
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
