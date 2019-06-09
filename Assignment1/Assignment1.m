clear,clc;
close all;

%% Read and Display Image
I=imread('cameraman.tif');
figure; imshow(I);

% Binarization
level=graythresh(I);  % Using Otsu thresholding
BW=imbinarize(I,level);
figure; imshow(BW);

%% Enhancement
I=imread('pout.tif');
EQ=histeq(I);

figure;
subplot(121);
imshow(I);
subplot(122);
imshow(EQ);

figure;
subplot(121);
imhist(I);
subplot(122);
imhist(EQ);

%% Denoising
I=imread('tire.tif');
% Add Gaussian Noise
M=0;
V=0.02;
Gaussian=imnoise(I,'gaussian',M,V);

% Add salt and pepper
d=0.06;
Salt_Pepper=imnoise(I,'salt & pepper',d);

figure;
subplot(121)
imshow(Gaussian);
subplot(122)
imshow(Salt_Pepper);

Denoise_G=medfilt2(Gaussian);
Denoise_S=medfilt2(Salt_Pepper);

figure;
subplot(121)
imshow(Denoise_G);
subplot(122)
imshow(Denoise_S);

%% Edge Detection
I=imread('cameraman.tif');
figure; imshow(I);

BW_P=edge(I,'Prewitt',[],'both');
BW_C=edge(I,'Canny');

figure;
subplot(121);
imshow(BW_P);
title('Perwitt');
subplot(122);
imshow(BW_C);
title('Canny');