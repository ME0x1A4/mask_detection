# mask_detection
Mask Detection for AI course

Task1:

The Task1 folder contains a matlab(2020a) script which will use resnet50 for image feature extraction and then train an SVM. It will also output a confusion matrix and the accuracy on the validation set. The 9943.mat contains all parameters and results of a training with the resulting accuracy of 99.43%. In other runs accuracy ranged from 99.14% - 99.43%.

To use the setup on a single picture use the following code:

% Create augmentedImageDatastore to automatically resize the image when
% image features are extracted using activations.
imageSize = net.Layers(1).InputSize;
ds = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');

% Extract image features using the CNN
imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');
Make a prediction using the classifier.

% Make a prediction using the classifier
predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')
