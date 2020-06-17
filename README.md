# mask_detection
Mask Detection for AI course

Task1:

The Task1 folder contains a matlab(2020a) script which will use resnet50 for image feature extraction and then train an SVM. It will also output a confusion matrix and the accuracy on the validation set. The 9943.mat contains all parameters and results of a training with the resulting accuracy of 99.43%. In other runs accuracy ranged from 99.14% - 99.43%.

To use the setup on a single picture use the following code:

% Create augmentedImageDatastore to automatically resize the image when image features are extracted using activations.

imageSize = net.Layers(1).InputSize;

ds = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');

% Extract image features using the CNN

imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');


% Make a prediction using the classifier

predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')



Ressource for Task1:

https://de.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html


Task2

The Task2 folder contains a python script to convert the labels/bounding boxes from the dataset to a matlab compatible file. The matlab face_detection.m script then uses the data to train a faster R-CNN detector. For this, images are resized to 224x224 RGB and the first layers of the pretrained resnet50 network are used to improve training accuracy and speed. Also data augumentation is used to increase training set size. 
It has to be noted that due to some unresoled bug the code cannot be executed on all matlab versions and further the labels/bounding boxes are probably still in a wrong format as it seems they are not detected correctly by matlab and result in a 0% accuracy training.

Ressource for Task2:

https://de.mathworks.com/help/vision/examples/object-detection-using-faster-r-cnn-deep-learning.html

