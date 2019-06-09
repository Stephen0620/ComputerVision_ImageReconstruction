clear,clc;
close all;

%% Ground truth
T=imread('teddyG.png');
T=double(T);
T=T./4;

%% Read Right and Left image
IL=imread('teddyL.png');
IL=rgb2gray(IL);
IR=imread('teddyR.png');
IR=rgb2gray(IR);
N=numel(IL);

%% SSD
window=[1,3,5,11,21];
maxDisp=64;
figure;
for i=1:numel(window)
    dispM{i}=findDisparity(IL,IR,maxDisp,window(i),'SSD');
    [dispM{i},~] = wiener2(dispM{i},[5,5]);
    dispM{i} = round(dispM{i});
    
    R=round(sqrt(sum(sum((dispM{i}-T).^2))/N),2);
    D=abs(dispM{i}-T);
    B=round(numel(find(D>1))/N,2);
    
    subplot(2,5,i);
    imshow(dispM{i},[]);
    title({['SSD win size:' num2str(window(i))],
        ['RMS: ' num2str(R) ' B: ' num2str(B)]});
    subplot(2,5,i+5);
    imshow(T,[]);
    title('ground truth');
end

%% SAD
window=[1,3,5,11,21];
maxDisp=64;
figure;
for i=1:numel(window)
    dispM{i}=findDisparity(IL,IR,maxDisp,window(i),'SAD');
    [dispM{i},~] = wiener2(dispM{i},[5,5]);
    dispM{i} = round(dispM{i});
    
    R=round(sqrt(sum(sum((dispM{i}-T).^2))/N),2);
    D=abs(dispM{i}-T);
    B=round(numel(find(D>1))/N,2);
    
    subplot(2,5,i);
    imshow(dispM{i},[]);
    title({['SAD win size:' num2str(window(i))],
        ['RMS: ' num2str(R) ' B: ' num2str(B)]});
    subplot(2,5,i+5);
    imshow(T,[]);
    title('ground truth');
end

%% NCC
window=[1,3,5,11,21];
maxDisp=64;
figure;
for i=1:numel(window)
    dispM{i}=findDisparity(IL,IR,maxDisp,window(i),'NCC');
    [dispM{i},~] = wiener2(dispM{i},[5,5]);
    dispM{i} = round(dispM{i});
    
    R=round(sqrt(sum(sum((dispM{i}-T).^2))/N),2);
    D=abs(dispM{i}-T);
    B=round(numel(find(D>1))/N,2);
    
    subplot(2,5,i);
    imshow(dispM{i},[]);
    title({['NCC win size:' num2str(window(i))],
        ['RMS: ' num2str(R) ' B: ' num2str(B)]});
    subplot(2,5,i+5);
    imshow(T,[]);
    title('ground truth');
end

%% Three dataset
Leftid={'teddyL.png','conesL.png','Art_L.png'};
Rightid={'teddyR.png','conesR.png','Art_R.png'};
Groundid={'teddyG.png','conesG.png','ArtG.png'};
for i=1:numel(Leftid)
    Left{i}=rgb2gray(imread(Leftid{i}));
    Right{i}=rgb2gray(imread(Rightid{i}));
    Ground{i}=double(imread(Groundid{i}))/4;
end

%% SSD with SAD
dispM=cell(3,1);
dispM_built=cell(3,1);
figure;
for i=1:numel(Left)
    dispM_built{i}=double(disparity(Left{i},Right{i},'Method','BlockMatching'));
    subplot(2,3,i+3);
    imshow(dispM_built{i},[]);
    dispM_built{i}(dispM_built{i}<0)=0;
    title('Built in function');
    
    dispM{i}=findDisparity(Left{i},Right{i},64,15,'SSD');
    R=round(sqrt(sum(sum((dispM{i}-dispM_built{i}).^2))/N),2);
    D=abs(dispM{i}-dispM_built{i});
    B=round(numel(find(D>1))/N,2);
    subplot(2,3,i);
    imshow(dispM{i},[]);
    title(['SSD: ' 'RMS:' num2str(R) ' B:' num2str(B)]);
    
end

%% SAD with SAD
dispM=cell(3,1);
dispM_built=cell(3,1);
figure;
for i=1:numel(Left)
    dispM_built{i}=double(disparity(Left{i},Right{i},'Method','BlockMatching'));
    subplot(2,3,i+3);
    imshow(dispM_built{i},[]);
    dispM_built{i}(dispM_built{i}<0)=0;
    title('Built in function');
    
    dispM{i}=findDisparity(Left{i},Right{i},64,15,'SAD');
    R=round(sqrt(sum(sum((dispM{i}-dispM_built{i}).^2))/N),2);
    D=abs(dispM{i}-dispM_built{i});
    B=round(numel(find(D>1))/N,2);
    subplot(2,3,i);
    imshow(dispM{i},[]);
    title(['SAD: ' 'RMS:' num2str(R) ' B:' num2str(B)]);
    
end

%% NCC with SAD
dispM=cell(3,1);
dispM_built=cell(3,1);
figure;
for i=1:numel(Left)
    dispM_built{i}=double(disparity(Left{i},Right{i},'Method','BlockMatching'));
    subplot(2,3,i+3);
    imshow(dispM_built{i},[]);
    dispM_built{i}(dispM_built{i}<0)=0;
    title('Built in function');
    
    dispM{i}=findDisparity(Left{i},Right{i},64,15,'NCC');
    R=round(sqrt(sum(sum((dispM{i}-dispM_built{i}).^2))/N),2);
    D=abs(dispM{i}-dispM_built{i});
    B=round(numel(find(D>1))/N,2);
    subplot(2,3,i);
    imshow(dispM{i},[]);
    title(['NCC: ' 'RMS:' num2str(R) ' B:' num2str(B)]);
    
end
