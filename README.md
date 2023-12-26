
# ASL-Letter-Classification
AI image classification algorithm of 26 American Sign Language Letters with Results

# Results

Image classification resulted in an accuracy of 98.77%, results for each character shown in Results.png.

# Setup

Download ASL Dataset (I got mine from kaggle.com). If not done already, separate images into test, train, and validation directories and then separate by letter within those directories. You can vary the number of images but the more the better the algorithm will work and you want ~90% of images in the training directory.

# Steps

(1) Prepare and separate image data

(2) Import and install necessary dependencies

(3) choose variables such as model type, number of epochs for training, and number of classes

(4) Write/tweak train_model, set_parameter_requires_grad, and initialize_model functions

(5) Initialize the model

(6) Do data augmentation and transformation for training and plot to verify it's working (Data_augmentation.png)

(7) Train and evaluate, assessing each epoch's accuracy and losses

(8) Train with scratch model and compare (Progress.png)

(9) Interpret data and faulties in algorithm (Results.png) -> try and improve for the future
