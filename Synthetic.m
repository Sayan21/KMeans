clear all;
K=10; pi = ones(K,1)/K;
N=5000; D=20; 
d=0.8; sig=1;
mu0= d*randn(K,D); X =zeros(N,D); trueLabel=zeros(N,1);
for i=1:N
    h=find(mnrnd(1,pi)==1);
    X(i,:) = mu0(h,:)+randn(1,D);
    trueLabel(i)=h;
end

% [C,IDX,Cost,T]=KMRand(X,K,.001);
% fprintf('K-Means NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
% [C,IDX,Cost,T]=KMPP(X,K,.001);
% fprintf('K-Means++ NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
% [C,IDX,Cost,T]=KMPL(X,K,.001);
% fprintf('K-Means|| NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));
% [C,IDX,Cost,T]=MoM(X,K,.001);
% fprintf('MoM: NMI: %f Rand Index: %f Purity: %f\n\n',nmi(IDX,trueLabel),RandIndex(IDX,trueLabel),Purity(IDX,trueLabel));

    [N D]=size(X);
    mX = sum(X,1)/N;X = X - repmat(mX,[N 1]);
    t0 = cputime;
    
    if(N>D)
        M2 = X'*X;
        [U,S]=eigs(M2,K); s = diag(S);
        W = U*diag(1./sqrt(s));
        
    else
        [P,S]=eigs(X*X',K);
        s=diag(S);
        W=X'*P*diag(1./s);
    end
    Mx=X*W;
    G = zeros(K,K,K);
    for i=1:K
        for j=1:K
            G(:,i,j)=(Mx(:,i).*Mx(:,j))'*Mx;
        end
        %fprintf(1,'Matrix Multiplied: %d\n',i);
    end
    
 
    Q=G; mu=0.01;
    for k=1:K
        Q(k,k,k)=Q(k,k,k)-mu;
    end
    
    v=unifrnd(0,1,[K,1]); v=v/sum(v);
    v=ones(K,1)/K;
    w0=randn(K,1); w0=w0./norm(w0);
    while 1
        fd=[];
        for k=1:K
            fd=[fd;w0'*Q(:,:,k)];
        end
        
        w=w0-2*fd\(fd*w0-v);
        fprintf(1,'Norm: %f\n',norm(w-w0));
        w0=w;
    end