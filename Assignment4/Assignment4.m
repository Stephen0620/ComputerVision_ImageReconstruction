clear,clc;
close all;

addpath('..\Assignment4\3DSIFT_CODE_v1');
%% Read video and 3d SIFT (Step 1 and 2)
% Path of the folder of videos
File = 'C:\Users\Stephen\Desktop\Computer Vision and Image Reconstruction\';
Action = {'run','jump','walk'}; %each actions has 9-10 videos

run = false;

if run ~= true
    load IVEC.mat;
else
    N=0;
    Keys = [];
    for i=1:numel(Action)
        Folder = [File Action{i}];
        A = dir(Folder); %open folder

        clear Name;
        for j=3:numel(A)
            Name{j-2} = A(j).name; % read video names
        end

        for j=1:numel(Name)% Read one video at a time 
            clear Frame;
            N = N + 1;
            fprintf('Start looking for the number %d Video\n',N);
            n=1;
            v = VideoReader([Folder '\' Name{i}]);
            while hasFrame(v)   % Create the 3D image, row, col, time
                Frame(:,:,n) = rgb2gray(readFrame(v));
                n=n+1;
            end

            clear keys
            
            for z=1:200 % Find 200 key points in the 3D image
                reRun = 1;
                while reRun == 1
                    % Randomly choose a point as a key point
                    r = randi([1,size(Frame,1)]);
                    c = randi([1,size(Frame,2)]);
                    t = randi([1,size(Frame,3)]);
                    loc = [r,c,t];
                    
                    fprintf(1,'Calculating keypoint at location (%d, %d, %d)\n',loc);

                    % Create a 3DSIFT descriptor at the given location
                    [keys{z} reRun] = Create_Descriptor(Frame,1,1,loc(1),loc(2),loc(3));
                end
            end
            % Each key point will generate a feature called ivec with
            % dimension equal to 1X640 (keys(z))
            % each video has 200 key points and each keypoint....
            for z=1:numel(keys)
                Keys = [Keys;keys{z}.ivec]; % concatinating keys for each video and each of theirs ivec
            end                             % 3 actiosn, 1 with 9 videos, 2 with 10 videos = 5800 key points and each key point 640 ivec points
        end
    end
end

%% Hierachical K-means (Step 3)
NCluster = 50; % Number of clusters
D = pdist(Keys); % pairwaise dist between keypoints
L = linkage(D); % Generate the linkage https://www.mathworks.com/help/stats/hierarchical-clustering.html
figure; dendrogram(L); title('Linkage');
T = cluster(L,'maxclust',NCluster); % clustering https://www.mathworks.com/help/stats/hierarchical-clustering.html

% Kmeans
C = zeros(NCluster,640); 
for i=1:NCluster
    idx = find(T==i);
    C(i,:) = mean(Keys(idx,:)); % Find centers of each cluster (mean)
end
[idx,C]= kmeans(double(Keys),NCluster,'Start',C);   % idx is cluster number

%% Signature Step 4
n=1; % here we use known part that 1st 200 points belong to 1st video and the idx that was the cluster center
       % each cluster is a word
for i=1:200:numel(idx)
    feats = Keys(i:i+200-1,:);
    index = idx(i:i+200-1,:);
    H(n,:) = histcounts(index,0.5:1:50.5);
    n=n+1;
end

% H gives signature of each video, i.e each value in matrix gives
% contribution of key points in that video... 29X50 ( i.e each row is
% signature of each video in every cluster)

%% Feature Grouping Histogram Step 5 
NClass = 50;
CoMAT = zeros(NClass,NClass);

for i=1:(n-1)
    temp = find(H(i,:)>0);
    coor = nchoosek(temp,2);
    for j=1:size(coor,1)
        CoMAT(coor(j,1),coor(j,2)) = CoMAT(coor(j,1),coor(j,2))+1; % calc features common to all 29 videos, hence useless in detection XXX
        CoMAT(coor(j,2),coor(j,1)) = CoMAT(coor(j,2),coor(j,1))+1;
    end
end

CorM = corr(CoMAT); % Calcultate the correlation of each words
CorM = triu(CorM) - diag(diag(CorM)); % Only need the upper triangle part
[row,col] = find(CorM>0.95);

GroupH = H;
% Grouping Histogram
for i=1:numel(row)
    GroupH(:,size(H,2)+i) = H(:,row(i))+H(:,col(i));
end
Temp = unique([row,col]);
GroupH(:,Temp) = [];

%% SVM Train with grouping histogram
Label = [ones(10,1);zeros(19,1)];
SVMmodel = fitcsvm(GroupH,Label,'Leaveout','on');
errorwithGroup = kfoldLoss(SVMmodel,'lossfun','classiferror');

%% SVM Train without grouping histogram
Label = [ones(10,1);zeros(19,1)];
SVMmodel = fitcsvm(H,Label,'Leaveout','on');
errorwithoutGroup = kfoldLoss(SVMmodel,'lossfun','classiferror');
