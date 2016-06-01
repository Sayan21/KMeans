clear all;
load('~/Downloads/CIFAR/cifar-100-matlab/train.mat');
X=double(data); 
clearvars data;

trueLabel = double(fine_labels)+1;
K=length(unique(trueLabel));
fprintf(1,'K = %d\n',K);

[C,IDX,Cost,T]=KMRand(X,K,.01);
fprintf('K-Means NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
[C,IDX,Cost,T]=KMPP(X,K,.01);
fprintf('K-Means++ NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
[C,IDX,Cost,T]=KMPL(X,K,.01);
fprintf('K-Means|| NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
[C,IDX,Cost,T]=MoM(X,K,.01);
fprintf('MoM: NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
