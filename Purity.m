function Purity=Purity(c1,c2)

K=max(c1); sumAg=0;
for k = 1:K
    sel = find(c1==k);
    tbl = tabulate(c2(sel));
    if ~isempty(sel)
        [~,I]=sort(tbl(:,2),'descend');
        sumAg=sumAg+tbl(I(1),2);
    end
end

Purity = sumAg/length(c1);
    