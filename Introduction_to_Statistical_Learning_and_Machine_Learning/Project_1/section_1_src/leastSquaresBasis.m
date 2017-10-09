function [model] = leastSquaresBasis(X, y, deg)

% generate the polynomial-form matrix
X_app = zeros(size(X, 1), deg + 1);
for i = 0 : deg
  X_app(:, i + 1) = X .^ i;
end

% calculate the estimated weight values
w_bias = (X_app' * X_app) \ X_app' * y;

% weight values with bias
model.w_bias = w_bias;
model.deg = deg;
model.predict = @predict;

end

function [y_hat] = predict(model, X)

% generate the polynomial-form matrix
X_app = zeros(size(X, 1), model.deg + 1);
for i = 0 : model.deg
  X_app(:, i + 1) = X .^ i;
end

% make prediction
y_hat = X_app * model.w_bias;

end