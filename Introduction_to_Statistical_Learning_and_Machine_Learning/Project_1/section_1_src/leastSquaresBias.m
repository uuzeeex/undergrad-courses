function [model] = leastSquaresBias(X, y)

% append a vector to the first column
X_app = [ones(size(X, 1), 1) X];

% calculate the estimated weight values
w_bias = (X_app' * X_app) \ X_app' * y;

% weight values with bias
model.w_bias = w_bias;
model.predict = @predict;

end

function [y_hat] = predict(model, X)

% append a vector to the first column
X_app = [ones(size(X, 1), 1) X];

% make prediction
y_hat = X_app * model.w_bias;

end