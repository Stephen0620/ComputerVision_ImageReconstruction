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
% Decode
N = 7;
X = linspace(-25,-15,N);
Y = linspace(10,-5,N);
% Temp = [X,Y];
Temp = zeros(N*N,2);
n=1;
for i=1:N
    for j=1:N
        Temp(n,:) = [X(i),Y(j)];
        n=n+1;
    end
end
w4probs = [Temp  ones(N*N,1)];
w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N*N,1)];
w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(N*N,1)];
w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(N*N,1)];
dataout = 1./(1 + exp(-w7probs*w8));

Draw = reshape(dataout',28,28,N,N);
figure;
n=1;
for i=1:N
    for j=1:N
        subplot(N,N,n);
        imshow(Draw(:,:,i,j),[]);
        n = n+1;
    end
end

% PCA
data = rawimages';
[coeff,score] = pca(data','Centered',false,'NumComponents',2);
figure; gscatter(coeff(:,1),coeff(:,2),ltest); title('PCA');