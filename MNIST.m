clear all;
X=[]; trueLabel=[];
filedir='~/Documents/MATLAB/Image/';
figure;
for i=0:9
    filename = sprintf('%s/digit%d.mat',filedir,i);
    load(filename);
    subplot(1,10,i+1);
    imshow(reshape(D(5,:),28,28)');
    X=[X;D]; trueLabel=[trueLabel; (i+1)*ones(size(D,1),1)];
end



K=length(unique(trueLabel));

[C,IDX,Cost,T]=KMRand(X,K,.001);
fprintf('K-Means NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
[C,IDX,Cost,T]=KMPP(X,K,.001);
fprintf('K-Means++ NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
[C,IDX,Cost,T]=KMPL(X,K,.001);
fprintf('K-Means|| NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
[C,IDX,Cost,T]=MoM(X,K,.001);
fprintf('MoM: NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
