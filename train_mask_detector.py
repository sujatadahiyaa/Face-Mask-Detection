# step 3:- DATA PREPROCESSING--->In this step I am going to convert all my from the both with mask or without mask dataset into arrays.
                                #So that without i can create a deploying model
# import the necessary packages

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os


# Initialize the initial learning rate, number of epochs to train for, and batch size.
INIT_LR = 1e-4  # learning rate must be least so that the loss rate can be calculated properly. It means I can get better accuracy. In this case, the learning rate is 0.0001
EPOCHS = 20
BS = 32

# Inside directory I mentioned where my dataset is present.
DIRECTORY = r"C:\Users\dahiy\OneDrive\Desktop\fmd project\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# Print to grab the list of images in our dataset directory, then initialize the list of data (i.e images) and class images
print("[INFO] loading images...")

data = []  # Inside this data empty list I am going to append all my image arrays inside this data list
labels = []  # Inside this label list I am going to append all those corresponding list which are with mask and without mask

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        # After preprocessing I add the image into the data list then add corresponding labels inside this labels list
        data.append(image)  # The data has numerical values
        labels.append(category)  # The labels have alphabetical values with mask or without mask

# Perform one-hot encoding on the labels
lb = LabelBinarizer()  # This comes from sklearn.preprocessing module. This LabelBinarizer method converts with mask or without mask into categorical variables
# i.e the code carries. It converts with mask and without mask into 0 and 1.
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# After successfully converting with mask and without mask into numerical values we need to convert them into numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Now I am going to use tester insplits to split my testing and training data. Here tester insplits are trainX, testX, trainY, testY. Here the test_size is 20%. So out
# of 1000 images 20% is given to the testing set and the rest of 80% will be for the training set

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

# Construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Using MobileNetV2 we are going to create two models. One is the MobileNet model whose output will be passing into the normal model that we are going to develop.
# We can call them as a head model and base model, respectively. (MobileNet model --> head model and normal model --> base model)

baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Construct the head of the model that will be placed on top of the base Model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)  # ReLU is basically an activation function for non-linear use cases. When we deal with images, we go for ReLU.
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)  # Here we go for our final output model. Output is 2 layers just because one is for with mask and without mask.
# In the output layer, we should go for softmax activation function or sigmoid activation function because they are probability-based 0 and 1 values.
# Since we are dealing with binary classification here, so I give softmax as the activation function

# Place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# Loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
# just because the replacement of convolution neural network model we are freezing them for training

for layer in baseModel.layers:
    layer.trainable = False

# Compile our model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,  # Adam optimizer is similar to ReLU that is used for any image prediction method
              metrics=["accuracy"])  # Here we are going to track accuracy metrics i.e the only matrix we are going to track

# Train the head of the network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# Make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)  # model.predict method is used to evaluate our network

# For each image in the testing set, we need to find the index of the label with the corresponding largest predicted probability for this we use np.argmax method
predIdxs = np.argmax(predIdxs, axis=1)

# Show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# Serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(r"C:\Users\dahiy\OneDrive\Desktop\fmd project\mask_detector_model.keras")


# Plot the training loss and accuracy using matplotlib
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
