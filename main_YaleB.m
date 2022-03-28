clear;clc; close all;
addpath './Datasets'
addpath './Utility'

load 'YaleB_32x32.mat'; lam = [.1 .5 25];% For Extended YaleB
fea = double(fea');
fea = fea./ repmat(sqrt(sum(fea.*fea)),[size(fea,1) 1]); % unit norm 2

train_num = 20;
acc =zeros(1,10);
rand('state',110);
for iter = 1
    %------------------------------------------------
    % Compute the number of samples per class
    %------------------------------------------------
    class_num = length(unique(gnd));
    numClass = zeros(class_num,1);
    for i=1:class_num
        numClass(i,1) = length(find(gnd==i));
    end
    
    %------------------------------------------------
    % Seperate train and test set
    %------------------------------------------------
    trfea = []; ttfea = []; 
    trgnd = []; ttgnd = [];
    for j = 1:class_num
        index = find(gnd == j); 
        randIndex = randperm(numClass(j));
        trfea = [trfea fea(:,index(randIndex(1:train_num)))];
        trgnd = [trgnd ; gnd(index(randIndex(1:train_num)))];
        numPerCls(j) = train_num;
        ttfea = [ttfea fea(:,index(randIndex(train_num+1:end)))];
        ttgnd = [ttgnd ; gnd(index(randIndex(train_num+1:end)))];
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
    para.lam1 = lam(1);
    para.lam2 = lam(2);
    para.lam3 = lam(3);
    para.dist = dist;
    para.numPerCls = numPerCls;
    [obj,Z,E] =  BDLRR(allFea, trfea, class_num,para);
    
    %------------------------------------------------
    % Final Recognition
    %------------------------------------------------
    [bstacc, bstlam] = recognition(Z,trgnd,ttgnd);
    fprintf('lam1=%.4f,lam2=%.4f,lam3=%.4f,acc = %.2f...\r\n',para.lam1,para.lam2,para.lam3,bstacc);
    acc(iter) = bstacc;
    disp(['++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++']);
    fprintf('The %d-th iter, classification acc=%.2f...\r\n',iter,bstacc);
end


