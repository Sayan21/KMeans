function [C,IDX,Cost,Time]=KMPP(X,K,epsilon)

    [N D]=size(X);
    mX = sum(X,1)/N;X = X - repmat(mX,[N 1]);
    t0 = cputime;
    
%% Mean Seeding    
    mu = zeros(K,D);
    z = unidrnd(size(X,1)); mu(1,:) = X(z,:); 

    dist = Inf*ones(size(X,1),1);
    for k=2:K
        randn('state',k);
        localDist = zeros(N,1);
        for idata=1:N
            localDist(idata)=(X(idata,:)-mu(k-1,:))*(X(idata,:)-mu(k-1,:))';

        end

        [dist,~] = min([dist,localDist],[],2);
        prob = dist/sum(dist);
        z = find(mnrnd(1,prob,1)==1); 
        mu(k,:) = X(z,:);
    end


 %% LLoyd's Iterations
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

    fprintf(1,'KMPP time %g iterations %d final cost %g\n',t1-t0,iter,sumDist(iter));
    