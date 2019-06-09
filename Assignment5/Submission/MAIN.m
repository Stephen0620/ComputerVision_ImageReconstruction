clear,clc;
close all;

addpath('..\Assignment5\Data');
addpath('.\codeD=2');
load('mnist_weights.mat')

%% Read
[rawimages,rawlabels] = readMNIST('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte');
rawimages = double(reshape(rawimages,28*28,size(rawimages,3)));
%% View 2D dimension reduction
N = 10000;
rawimages = rawimages(:,1:N);
data = [rawimages' ones(N,1)];
w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs ones(N,1)];
w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs ones(N,1)];
w4probs = w3probs*w4;
ltest = rawlabels(1:N);
figure; gscatter(w4probs(:,1),w4probs(:,2),ltest); title('Autoencoder');

%% Generate Number
rect = getrect;
r = rect(1) + (rect(1)+rect(3)-rect(1))*rand(1,1);
c = rect(2)-rect(4) + (rect(2)-(rect(2)-rect(4)))*rand(1,1);

% Decode
w4probs = [[r,c]  ones(1,1)];
w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(1,1)];
w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(1,1)];
w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(1,1)];
dataout = 1./(1 + exp(-w7probs*w8));

Draw = reshape(dataout,28,28);
figure; imshow(Draw,[]);

% PCA
data = rawimages';
[coeff,score] = pca(data','Centered',false,'NumComponents',2);
figure; gscatter(coeff(:,1),coeff(:,2),ltest); title('PCA');