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
I = rgb2gray(I);
[~,features,points] = sift(I);

I2 = readimage(Scene,2);
I2 = rgb2gray(I2);
[~,features2,points2] = sift(I2);

% Match features
[num,idx1,idx2] = match(I,I2);

% Swapping the columns
idx1(:,[1 2]) = idx1(:,[2 1]);
idx2(:,[1 2]) = idx2(:,[2 1]);

figure;
ax = axes;
showMatchedFeatures(I,I2,idx1(:,1:2),idx2(:,1:2),'montage','Parent',ax);
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
    [~,features,points] = sift(grayImage);

    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);

    matchedPoints = points(indexPairs(:,1), 1:2);
    matchedPoints(:,[1 2]) = matchedPoints(:,[2 1]);  % Swapping the columns
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), 1:2);
    matchedPointsPrev(:,[1 2]) = matchedPointsPrev(:,[2 1]);  % Swapping the columns

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