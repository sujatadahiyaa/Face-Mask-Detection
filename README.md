# FACE MASK DETECTION SYSTEM

This project is developed using Python and Machine Learning to detect whether a person is wearing the face mask or not. It uses a combination of computer vision or convolutional neural networks(CNNs) to perform real-tme mask detection from images or video streams. 

# LIBRARIES AND TOOLS USED:

1. TensorFlow: It is an open-source deep-learning framework used for building and training CNN model. It provides flexibbility in model training, evaluation and deployment.
2. Keras: A high-level neural networks API running on top of TenserFlow. It makes model building easy with its user-friendly and modular interface. it is used to define the CNN architecture.
3. OpenCV (Open Source Computer Vision Library): It is used for image and video capture, as well as face detection using Haar Cascades or deep-learning-based detectors. It helps in real-time video processing from webcam or CCTV feeds.
4. MobileNetV2: A pre-trained lightweight deep learning model optimized for mobile and edge devices. It is used for feature extraction(transfer learning) to improve accuracy and reduce training time.
5. Numpy and Pandas: These are used for data manipulation and numerical computations. These are very useful in loading datasets and preprocessing image arrays.
6. Matplotlib and Seaborn: These libraries are used for visualizing training performance like accuracy and loss graphs.

# How it works:-

A) Dataset Collection
1. The dataset consists of images of people with or without face masks.
2. Publicly available datasets like the "FACE MASK DATASET" from Kaggle is used.
3. Images are labeled in two categories: with_mask and without_mask.
4. The size of dataset is 3833 items i.e 1915(with_mask) and 1918(without_mask).

B) Data Preprocessing
1. Resize all the images to a fixed size(e.g., 224*224 pixcels).
2. Convert images to arrays using OpenCV.
3. Normalize pixcel values to the range[0,1].
4. Split data into traing and validation sets.

C) Model Building using MobileNetV2
1. Load the MobileNetV2 model with include_top=False to remove the classification head.
2. Freeze the base layers of MobileNetV2 to retain pre-trained weights.
3. Add custom layers:
         a) GlobalAveragePooling2D
         b) Dense (ReLU)
         c) Dropout (for regularization)
         d) Final Dense layer with softmax activation for binary classification.

D) Model Compilation and Training
1. Compile the model with:
         a) Loss Function: binary_crossentropy
         b) Optimizer: Adam
         c) Metrics: accuracy
2. Train the model on the dataset for several epochs( e.g., 10-20) with appropriate batch size.

E) Model Evaluation
1. Evaluate the model on the validation set.
2. Plot acciracy and loss curves to check for overfitting or underfitting.

F) Real-Time Face Mask Detection
1. Use OpenCV to capture video frames from webcam.
2. Detect faces using a face detector (e.g., Haar Cascade or DNN).
3. Extract the face ROI(rregion of interest), preprocess it, and pass it to the trained model.
4. Display the prediction(e.g., "Mask" or "No Mask" with percentage how accurately the face is covered) on the screen with bounding boxes.

G) Deployment
1. The model can be deployed using:
           a) A desktop GUI app using tkinter or PyQT.
           b) A web app using Flask.
           c) Integrated with Raspberry Pi and cameras for embeded use in public places.

# Applications

1. Airports, Railway Stations, Shopping Malls.
2. Office buildings and entry gates.
3. Health Monitoring systems in smart cities.
4. Automated alert system for non-compilance.

# Key Highlights

Accuracy: It can achieve over 95% accuracy with a well-prepared dataset.
Speed: Optimized for real-time performance using MobileNetV2.
Scalability: It can be deployed on various platforms - desktop, web or edge devices.

# By pressing "q" we can break the loop. The system will be closed.
   

  
   
