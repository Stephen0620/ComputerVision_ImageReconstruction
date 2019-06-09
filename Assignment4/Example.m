clear,clc;
close all;

%%
X = [1 2;2.5 4.5;2 2;4 1.5;4 2.5];
figure; scatter(X(:,1),X(:,2),'rx','LineWidth',1.5);

Y = pdist(X);
squareform(Y)

Z = linkage(Y);
dendrogram(Z)

T = cluster(Z,'maxclust',3);

for i=1:3
    idx = find(T==i);
    C(i,:) = mean(X(idx,:));
end
C(3,:) = X(2,:);
figure; scatter(X(:,1),X(:,2),'rx','LineWidth',1.5);
hold on;
scatter(C(:,1),C(:,2),'bo','LineWidth',1.5);
hold off;