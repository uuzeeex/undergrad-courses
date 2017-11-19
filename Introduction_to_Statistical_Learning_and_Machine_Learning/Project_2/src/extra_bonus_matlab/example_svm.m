load quantum.mat
[n,d] = size(X);

% Split into training and validation set
perm = randperm(n);
Xvalid = X(n / 2 + 1 : end, :);
yvalid = y(n / 2 + 1 : end);
X = X(1 : n / 2, :);
y = y(1 : n / 2);

n = n / 2;
lambda = 1 / n;
model = svmSAG(X, y, lambda, 80 * n);