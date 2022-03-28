function [obj,Z,E] = BDLRR(X, X_bar, clsNum,para)
numPerCls = para.numPerCls;
lam1 = para.lam1;
lam2 = para.lam2;
lam3 = para.lam3;
dist = para.dist;
%------------------------------------------------
% Paramters initialization
%------------------------------------------------
[d,n] = size(X);
[~,m] = size(X_bar);
tol = 1e-6;
maxIter = 1e4;
rho = 1.15;
mu = sqrt(max(d,m))\1;
max_mu = 1e8;
Z = zeros(m,n);
E = zeros(size(X));
P = zeros(m,n);
Q = zeros(m,n);
R = zeros(m,n);
C1 = zeros(d,n);
C2 = zeros(m,n);
C3 = zeros(m,n);
XTX = X_bar'*X_bar;
%% Start main loop
iter = 0;
while iter<maxIter
    iter = iter + 1;
    temp = Z - C2/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    
    %------------------------------------------------
    % Update P
    %------------------------------------------------
	svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    P = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    clear U V sigma svp temp;
    
    %------------------------------------------------
    % Update Z
    %------------------------------------------------ 
    Z_left = XTX+( lam1/mu+2)*eye(m);
    Z = Z_left \ ( X_bar'*(X-E)+P+Q+(X_bar'*C1 + C2 + C3 + lam1*R)/mu);
    clear Z_left;
    
    %------------------------------------------------
    % Update Q
    %------------------------------------------------
    B = mu\lam2*dist;
    temp = Z-C3/mu;
    Q = solve_l1_norm(temp,B);
    clear B temp;
    
    %------------------------------------------------
    % Update E
    %------------------------------------------------
    temp = X-X_bar*Z+C1/mu;
    E = solve_l1l2(temp,lam3/mu);
    clear temp;
    
    %------------------------------------------------
    % Extract Block-Diagonal Components (R)
    %------------------------------------------------
    Z_block = cell(clsNum,1);
    for k = 1:clsNum
        range = sum(numPerCls(1:k-1))+1:sum(numPerCls(1:k));
        Z_block{k} = Z(range,range);        
    end
    R = [blkdiag(Z_block{:}), zeros(m, size(X,2)-m)];
    clear Z_block;
    
    %------------------------------------------------
    % Convergence Validation
    %------------------------------------------------
    leq1 = X-X_bar*Z-E;
    leq2 = P-Z;
    leq3 = Q-Z;
    stopC1 = max(max(max(abs(leq1))),max(max(abs(leq2))));
    stopC = max(stopC1,max(max(abs(leq3))));
    if stopC<tol || iter>=maxIter
        break;
    else
        C1 = C1 + mu*leq1;
        C2 = C2 + mu*leq2;
        C3 = C3 + mu*leq3;
        mu = min(max_mu,mu*rho);
    end
    if (iter==1 || mod(iter, 5 )==0 || stopC<tol)
            disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    obj(iter) = norm(leq1,'fro')/norm(X,'fro');
end
end

% Soft thresholding
function [E] = solve_l1_norm(x,varepsilon)
     E = max(x- varepsilon, 0);
     E = E+min( x+ varepsilon, 0);   
end

% Solving L_{21} norm minimization
function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end
