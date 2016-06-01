load('~/Downloads/Images_KMeans/yaleB.mat');
X=double(data.images); trueLabel = data.labels; 
K=length(unique(trueLabel));
clearvars data;

[C,IDX,Cost,T]=KMRand(X,K,.01);
fprintf('K-Means NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
[C,IDX,Cost,T]=KMPP(X,K,.01);
fprintf('K-Means++ NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
[C,IDX,Cost,T]=KMPL(X,K,.01);
fprintf('K-Means|| NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
[C,IDX,Cost,T]=MoM(X,K,.01);
fprintf('MoM: NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
