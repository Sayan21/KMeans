clear all;
X=[]; trueLabel=[];
%figure;
for i=1:5
    filename = sprintf('~/Downloads/CIFAR/cifar-10-batches-mat/data_batch_%d.mat',i);
    load(filename);
    %subplot(1,10,i+1);
    X=[X;double(data)]; trueLabel=[trueLabel; double(labels)];
end
trueLabel=trueLabel+1;
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
