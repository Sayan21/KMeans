function [C,IDX,Cost,Time]=KMPL(X,K,epsilon)

    [N D]=size(X);
    mX = sum(X,1)/N;X = X - repmat(mX,[N 1]);
    t0 = cputime;
    mu = zeros(K,D);

    z = unidrnd(size(X,1));
    dist = sum((X-repmat(X(z,:),[N 1])).*(X-repmat(X(z,:),[N 1])),2);

    
    R=5; L=2*K;
    C = X(z,:);
    for r=1:R
        prob = dist/sum(dist);
        C1=zeros(L,D);
        for l=1:L
            z = find(mnrnd(1,prob,1)==1);
            C1(l,:)=X(z,:);
        end
        fprintf(1,'l=%d ',r); 
        localDist = zeros(N,L);
        for l=1:L
            localDist(:,l)=sum((X-repmat(C1(l,:),[N 1])).*(X-repmat(C1(l,:),[N 1])),2);
        end
        [~,label] = min(localDist,[],2);

        [dist,~] = min([dist,localDist(label)],[],2);

        C=[C;C1];

    end
    fprintf(1,'\n');
    dist=zeros(N,size(C,1));
    for k=1:size(C,1)
        dist(:,k) = sum((X-repmat(C(k,:),[N 1])).*(X-repmat(C(k,:),[N 1])),2);
    end
    [~,label] = min(dist,[],2);
    clearvars dist;
    w=zeros(size(C,1),1);
    for k=1:size(C,1)
        w(k)=length(find(label==k)+.1)/N;
    end

%======================== Weighted K-Means =============================%
    nIter=10; freq = zeros(K,1); sumDist=zeros(nIter,1); prevSumDist=100; 
    mu = randn(K,D);
    for iter=1:nIter
        dist=zeros(size(C,1),K);
        for k=1:K
            dist(:,k)=w.*sum((C-repmat(mu(k,:),[size(C,1) 1])).*(C-repmat(mu(k,:),[size(C,1) 1])),2);
        end
        [~,label] = min(dist,[],2);

        sumDist(iter)=0;
        for k=1:K
            sel = find(label==k);

            if ~isempty(sel)
                freq(k) = length(sel);
                mu(k,:) = sum(C(sel,:),1)/freq(k);
            end
        end
        sumDist(iter)=sum(w.*sum((C-mu(label,:)).*(C-mu(label,:)),2),1);
        if abs(prevSumDist-sumDist(iter)) < epsilon*prevSumDist
            break;
        end
        prevSumDist = sumDist(iter);

        fprintf(1,'Weighted K-Means for Seeding %dth iter: %f\n',iter,sumDist(iter));
    end


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

    fprintf(1,'KMPL time %g iterations %d final cost %g\n',t1-t0,iter,sumDist(iter));
    