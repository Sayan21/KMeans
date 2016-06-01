function [C,IDX,Cost,Time]=MoM(X,K,epsilon)

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

    G=tensor(G);
    eigvals = zeros(K,1);
    V=zeros(K,K);
     for k=1:K
        G=symmetrize(G);
        [s,U]=sshopm(G);
        if(s>0) eigvals(k)=s; V(:,k)=U;
        else eigvals(k)=-s; V(:,k)=-U;
        end

        G=G-tensor(ktensor(s,U,U,U));
        fprintf(1,'%dth EigenValue Extracted: %f\n',k,s);
    end

    clearvars G;
    V2 = zeros(D,K); W2 = pinv(W'); pi = 1./(eigvals.*eigvals); 
    z = sum(pi);
    for k=1:K
        V2(:,k)=eigvals(k)*W2*V(:,k);
    end
    mu = V2';

    t_mom = cputime;

    fprintf('Time for MoM: %g\n',t_mom-t0);


    nIter=50;
    freq = zeros(K,1); sumDist=zeros(nIter,1); prevSumDist=100;
    
    if(N>D)
        for iter=1:nIter
            dist=zeros(N,K);
            for k=1:K
                dist(:,k)=sum((X-repmat(mu(k,:),[N 1])).*(X-repmat(mu(k,:),[N 1])),2);
            end
            [~,label] = min(dist,[],2);


            for k=1:K
                sel = find(label==k);

                if ~isempty(sel)
                    freq(k) = length(sel);
                    mu(k,:) = sum(X(sel,:),1)/freq(k);
                end
            end
            sumDist(iter)=sum(sum((X-mu(label,:)).*(X-mu(label,:)),2),1);
            fprintf(1,'%d iter: %f\n',iter,sumDist(iter));
            if abs(prevSumDist-sumDist(iter)) < epsilon*prevSumDist
                break;
            end
            prevSumDist = sumDist(iter);

        end
    else
        dist=zeros(N,K);
        for k=1:K
            dist(:,k)=sum((X-repmat(mu(k,:),[N 1])).*(X-repmat(mu(k,:),[N 1])),2);
        end
        [~,label] = min(dist,[],2);
        iter=1;
        sumDist(iter)=sum(sum((X-mu(label,:)).*(X-mu(label,:)),2),1);
    end
        


    t1 = cputime;
    IDX=label;
    C=mu+repmat(mX,[K 1]);
    Cost=sumDist(iter);
    Time = t1-t0;

    fprintf(1,'MoM time %g iterations %d final cost %g\n',t1-t0,iter,sumDist(iter));
    