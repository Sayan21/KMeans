function [C,IDX,Cost,Time]=KMRand(X,K,epsilon)

    [N D]=size(X);
    mX = sum(X,1)/N;X = X - repmat(mX,[N 1]);
    t0 = cputime;
    mu = normrnd(0,1,[K,D]);

    nIter=50;
    freq = zeros(K,1); sumDist=zeros(nIter,1); prevSumDist=100;
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

    t1 = cputime;
    IDX=label;
    C=mu+repmat(mX,[K 1]);
    Cost=sumDist(iter);
    Time = t1-t0;

    fprintf(1,'KMeans time %g iterations %d final cost %g\n',t1-t0,iter,sumDist(iter));
    