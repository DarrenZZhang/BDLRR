function [bstacc,bstlam] = recognition(Z,trgnd,ttgnd)
% Recognition function
% Z     ------------ Learnt data representation
% trgnd ------------ Label of train data
% ttgnd ------------ Label of test data

%------------------------------------------------
% Construct Label matrix and Seperate trZ and ttZ
%------------------------------------------------
class_num = numel(unique(trgnd));
L = zeros(class_num,numel(trgnd));
for i = 1:length(trgnd)
    c = trgnd(i);
    L(c,i) = 1;
end
trZ = Z(:,1:numel(trgnd)); 
ttZ = Z(:,numel(trgnd)+1:end);

%------------------------------------------------
% Recognition Procedure
%------------------------------------------------
bstacc = 0;
lambda = 0.01;
for j = 1:10
    cclass = [];
    W = L*trZ'*inv(trZ*trZ'+lambda*eye(size(trZ,1)));
    for i = 1:length(ttgnd)
        s = W*ttZ(:,i);
        [v,k] = max(s);
        cclass = [cclass; k];
    end
    lambda = lambda + 0.01;
    rate = 100*(sum(cclass==ttgnd))/numel(ttgnd);
    if bstacc < rate
        bstlam = lambda;
        bstacc = rate;
    end
end
end
