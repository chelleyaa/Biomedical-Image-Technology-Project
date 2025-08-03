%% Set up the environment
clc;
clear;
close all;

%% Load and display the image dataset
datasetFolder = 'Alzheimer Dataset';
imds = imageDatastore(datasetFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% Balance the dataset by limiting each class to 600 images
% Count each label to find the minimum number of images in any label
labelCount = countEachLabel(imds);
minCount = min(labelCount.Count);

% Determine the number of images per label to use for balancing
numImagesPerLabel = min(minCount, 600);

% Balance the dataset by limiting each class to numImagesPerLabel images
imds = splitEachLabel(imds, numImagesPerLabel, 'randomized');

%% Count each label
labelCount = countEachLabel(imds);

% Display the number of images per category
disp(labelCount);

%% Preprocess the images: filtering, CLAHE, smoothing, and feature extraction
function [Iout, features] = preprocessImage(I)
    % Convert to grayscale if the image is RGB
    if size(I, 3) == 3
        I = rgb2gray(I);
    end

    % Apply median filtering to remove noise
    Ifiltered = medfilt2(I, [3 3]);
    
    % Apply CLAHE to enhance the contrast
    Iclahe = adapthisteq(Ifiltered);
    
    % Apply Gaussian smoothing to reduce noise and smooth the image
    Ismoothed = imgaussfilt(Iclahe, 2);
    
    % Extract HOG features
    features = extractHOGFeatures(Ismoothed);
    
    Iout = Ismoothed;
end

% Apply preprocessing and extract features
imdsTransformed = transform(imds, @(x) preprocessImage(x));

% Extract and display unique features for each class
labelNames = unique(imds.Labels);
featuresPerClass = cell(numel(labelNames), 1);
for i = 1:numel(labelNames)
    classImages = subset(imds, imds.Labels == labelNames(i));
    numImages = min(10, numel(classImages.Files)); % Use up to 10 images per class for feature extraction
    features = [];
    for j = 1:numImages
        [~, feature] = preprocessImage(readimage(classImages, j));
        features = [features; feature];
    end
    featuresPerClass{i} = features;
    disp(['Features for class ', char(labelNames(i)), ':']);
    disp(features);
end

%% Visualize original and preprocessed images
figure;
subplot(1,3,1);
imshow(readimage(imds,1));
title('Original Image');

subplot(1,3,2);
[preprocessedImage, ~] = preprocessImage(readimage(imds,1));
imshow(preprocessedImage);
title('Preprocessed Image');

%% Split dataset into training and validation sets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

%% Load pretrained Alexnet network
net = alexnet;

% Adjust the network for transfer learning
lgraph = layerGraph(net);

% Replace the classification layer with new layers
numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_classoutput')];

lgraph = replaceLayer(lgraph, 'fc8', newLayers(1));
lgraph = replaceLayer(lgraph, 'prob', newLayers(2));
lgraph = replaceLayer(lgraph, 'output', newLayers(3));

%% Resize the images to the input size of the network
inputSize = net.Layers(1).InputSize;

% Define image data augmenter
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10,10], ...
    'RandXTranslation', [-3 3], ...
    'RandYTranslation', [-3 3]);

augmentedTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', imageAugmenter);
augmentedValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

%% Set training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augmentedValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Train the network
trainedNet = trainNetwork(augmentedTrain, lgraph, options);

%% Evaluate the trained network
YPred = classify(trainedNet, augmentedValidation);
YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation);
disp(['Validation Accuracy: ', num2str(accuracy)]);

%% Confusion matrix
figure;
cm = confusionchart(YValidation, YPred);
cm.Title = 'Confusion Matrix';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%% Calculate precision, recall, and F1 score
precision = diag(cm.NormalizedValues) ./ sum(cm.NormalizedValues, 2);
recall = diag(cm.NormalizedValues) ./ sum(cm.NormalizedValues, 1)';
f1score = 2 * (precision .* recall) ./ (precision + recall);

fprintf('Precision: %.2f\n', mean(precision));
fprintf('Recall (Sensitivity): %.2f\n', mean(recall));
fprintf('F1 Score: %.2f\n', mean(f1score));
