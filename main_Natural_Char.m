clear;clc; close all;
addpath './Datasets'
addpath './Utility'

% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Important Note: All the scene character features can ba loaded 
% from the homepage of Prof. Yong Xu: http://www.yongxu.org/databases.html
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

load 'Char74k_15_HOG.mat'
trfea = trfea./ repmat(sqrt(sum(trfea.*trfea)),[size(trfea,1) 1]); % unit norm 2
ttfea = ttfea./ repmat(sqrt(sum(ttfea.*ttfea)),[size(ttfea,1) 1]); % unit norm 2

%------------------------------------------------
% Compute the number of samples per class
%------------------------------------------------
class_num = length(unique(trgnd));
numPerCls = zeros(class_num,1);
for i=1:class_num
    numPerCls(i,1) = length(find(trgnd==i));
end

%------------------------------------------------
% Calculate Distance
%------------------------------------------------
allFea = [trfea ttfea];
dist = L2_distance(trfea,allFea);
dist = exp(dist-repmat(max(dist,[],2),[1 size(dist,2)])); % Normalize to [0,1]

%------------------------------------------------
% Learn Block-diagonal Representation
%------------------------------------------------
lam = [5 1 10];% Char74K,SVT
para.lam1 = lam(1);
para.lam2 = lam(2);
para.lam3 = lam(3);
para.dist = dist;
para.numPerCls = numPerCls;
[obj,Z,E] =  BDLRR(allFea, trfea, class_num,para);

%------------------------------------------------
% Construct Label matrix and Seperate trZ and ttZ
%------------------------------------------------
L = zeros(class_num,size(trfea,2));
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
    rate = 100*(sum(cclass==ttgnd))/size(ttfea,2);
    if bstacc < rate
        bstlam = lambda;
        bstacc = rate;
    end
end
disp(['+++++++++++++++++++++++++++++Final Results+++++++++++++']);
fprintf('lam1=%.4f,lam2=%.4f,lam3=%.4f,acc = %.2f...\r\n',para.lam1,para.lam2,para.lam3,bstacc);

