clear,clc;
close all;
addpath('.\Data');

[Images,~] = readMNIST('train-images-idx3-ubyte','train-labels-idx1-ubyte');
[TestImages,~] = readMNIST('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte');

batchSize = 100;
Digit = Images(:,:,1:batchSize);

X = double(reshape(Digit,[784,batchSize]));
X = X./255;
t = X;

D = size(X,1);
Layers = [D,2000,1000,500,30,500,1000,2000,D]; % Number of units in each layers, input , hidden and output
 
%Activation Function
H{1} = @(x) 1./(1+exp(-x));
H{2} = @(x) 1./(1+exp(-x)); % Output Layer activation function
H{3} = @(x) 1./(1+exp(-x));
H{4} = @(x) 1./(1+exp(-x));
H{5} = @(x) 1./(1+exp(-x));
H{6} = @(x) 1./(1+exp(-x));
H{7} = @(x) 1./(1+exp(-x));
H{8} = @(x) 1./(1+exp(-x));
dH{1} = @(x) exp(-x)./((exp(-x) + 1).^2);
dH{2} = @(x) exp(-x)./((exp(-x) + 1).^2);
dH{3} = @(x) exp(-x)./((exp(-x) + 1).^2);
dH{4} = @(x) exp(-x)./((exp(-x) + 1).^2);
dH{5} = @(x) exp(-x)./((exp(-x) + 1).^2);
dH{6} = @(x) exp(-x)./((exp(-x) + 1).^2); 
dH{7} = @(x) exp(-x)./((exp(-x) + 1).^2); 
dH{8} = @(x) exp(-x)./((exp(-x) + 1).^2); 

%% Feed Forward Network
Z{1} = X;
W = cell(numel(Layers)-1,1);    % Weight Vector
B = cell(numel(Layers)-1,1);    % Bias term
Delta = cell(numel(Layers)-1,1);
rho = 0.01;

% Initial Guess Might need to be revised
for i=1:numel(W)
%     W{i} = 0.01+(0.1-0.01)*rand(Layers(i+1),Layers(i));
%     b = 0.01+(0.1-0.01)*rand(Layers(i+1),1);
    W{i} = rand(Layers(i+1),Layers(i));
    b = zeros(Layers(i+1),1);
    B{i} = repmat(b,[1,size(X,2)]);
end
 
Epoch = 30;
for j=1:Epoch
    % Feed Forward
    for i=1:numel(Layers)-1
        A{i} = W{i}*Z{i}+B{i};
        h = H{i};
        Z{i+1} = h(A{i});
    end
    y = Z{numel(Layers)};
     
    % Back propagation
    Delta{end} = (y-t); % Output layer cost derivative
%     E(j) = sum(-t.*log(y)-(1-t).*log(1-y)); % Cost function
%     E(j) = sum((y-t).^2); % Cost function
    for i=numel(Layers)-1:-1:1
        if i==1
            break;
        else
            dh = dH{i-1};
            Delta{i-1}  = W{i}'*Delta{i}.*dh(A{i-1});    % Delta(1)
        end
    end    
    for i=numel(Layers)-1:-1:1
        W{i} = W{i}-rho.*(Delta{i}*Z{i}');
        b = B{i}(:,1)-rho.*sum(Delta{i},2);
        B{i} = repmat(b,[1,size(X,2)]);
    end
end
errTrain = (1/batchSize).*sum(sum( (X-Z{end}).^2 ))

%% Reconstruct
Output = Z{end};
Output = reshape(Output,28,28,batchSize);

Original = reshape(X,28,28,batchSize);
Show = [Original(:,:,1);Output(:,:,1)];
figure; imshow(Show,[]);

%% Test Data
Test = double(TestImages(:,:,1))./255;
Temp = reshape(Test,[numel(Test),1]);

clear Z A;
Z{1} = Temp;
for i=1:numel(B)
    B{i} = B{i}(:,1);
end
% FeedFoward;
for i=1:numel(Layers)-1
    A{i} = W{i}*Z{i}+B{i};
    h = H{i};
    Z{i+1} = h(A{i});
end

OutputTest = reshape(Z{end},28,28);
errTest = sum(sum( (Temp-Z{end}).^2 ))

Show = [Test;OutputTest];
figure; imshow(Show,[]);