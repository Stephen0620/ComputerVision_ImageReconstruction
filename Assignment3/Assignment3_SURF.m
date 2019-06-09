clear,clc;
close all;

%% Importing Images
% Load images.
imageDir = fullfile('C:\Users\Stephen\Dropbox\MATLAB\Computer Vision and Image Reconstruction\Assignment3\images');
% imageDir = fullfile(toolboxdir('vision'), 'visiondata', 'building');
Scene = imageDatastore(imageDir);

% Display images to be stitched
montage(Scene.Files);

%% Feature Detection
I = readimage(Scene, 1);
grayImage = rgb2gray(I);
Threshold=1000;
points = detectSURFFeatures(grayImage,'MetricThreshold',Threshold);
[features, points] = extractFeatures(grayImage, points);

figure; imshow(I); hold on; plot(points);

I2 = readimage(Scene, 2);
grayImage = rgb2gray(I2);
points2 = detectSURFFeatures(grayImage);
[features2, points2] = extractFeatures(grayImage, points2);

% Match features
indexPairs = matchFeatures(features, features2, 'Metric', ...
    'SAD', 'MatchThreshold', 5);
matchedPoints1 = points(indexPairs(:,1),:);
matchedPoints2 = points2(indexPairs(:,2),:);

figure;
showMatchedFeatures(I, I2, matchedPoints1, matchedPoints2);
legend('matched points in I1', 'matched points in I2');

%% Estimate geometric transformation
numImages = numel(Scene.Files);
tforms(numImages) = projective2d(eye(3));

% Initialize variable to hold image sizes.
imageSize = zeros(numImages,2);

% Iterate over remaining image pairs
for n = 2:numImages

    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;

    % Read I(n).
    I = readimage(Scene, n);

    % Convert image to grayscale.
    grayImage = rgb2gray(I);

    % Save image size.
    imageSize(n,:) = size(grayImage);

    % Detect and extract SURF features for I(n).
    points = detectSURFFeatures(grayImage,'MetricThreshold',Threshold);
    [features, points] = extractFeatures(grayImage, points);

    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);

    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);

    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);

    % Compute T(n) * T(n-1) * ... * T(1)
    tforms(n).T = tforms(n).T * tforms(n-1).T;
end

%% Find the center image
% Compute the output limits  for each transform
for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

avgXLim = mean(xlim, 2);
[~, idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx)
Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)
    tforms(i).T = tforms(i).T * Tinv.T;
end

%% Initialize the Panorama
for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);

%% Create the Panorama
blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:numImages
    I = readimage(Scene, i);

    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

    % Generate a binary mask.
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);

    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
end

figure;
imshow(panorama);