clear all;
load iris_dataset; X=irisInputs'; trueLabel=vec2ind(irisTargets)';
K=length(unique(trueLabel));

N=size(X,1); D=size(X,2); Sig = std(X,1);
mX = sum(X,1)/N;


Fea = zeros(N,D*(D-1)/2);
count=1;
for i=1:D
    for j=i+1:D
        Fea(:,count)=(X(:,i)-repmat(mX(i),[N 1])).*(X(:,j)-repmat(mX(j),[N 1]))/sqrt(Sig(i)*Sig(j));
        count=count+1;
    end
end

X=[X,Fea]; D=D+size(Fea,2);

[C,IDX,Cost,T]=KMRand(X,K,.01);
fprintf('K-Means NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
[C,IDX,Cost,T]=KMPP(X,K,.01);
fprintf('K-Means++ NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
[C,IDX,Cost,T]=KMPL(X,K,.01);
fprintf('K-Means|| NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
[C,IDX,Cost,T]=MoM(X,K,.01);
fprintf('MoM: NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
