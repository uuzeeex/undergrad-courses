function [model] = svmAvg(X, y, lambda, maxIter)

% Add bias variable
[n,d] = size(X);
X = [ones(n,1) X];

% Matlab indexes by columns,
%  so if we are accessing rows it will be faster to use  the traspose
Xt = X';

% Initial values of regression parameters
w = zeros(d + 1, 1);
w_set = []
% Apply stochastic gradient method
for t = 1 : maxIter

    if t >= maxIter / 2
        w_set = [w_set, w];
    end

    if mod(t - 1, n) == 0
        % Plot our progress
        % (turn this off for speed)
        if t >= maxIter / 2
            w_ = mean(w_set, 2);
        else
            w_ = w
        end
        objValues(1 + (t - 1) / n) = (1 / n) * sum(max(0, 1 - y .* (X * w_))) + (lambda / 2) * (w_' * w_);
        semilogy([0 : t / n], objValues);
        pause(.1);
    end
    
    % Pick a random training example
    i = ceil(rand * n);
    
    % Compute sub-gradient
    [~, sg] = hingeLossSubGrad(w, Xt, y, lambda, i);
    
    % Set step size
    alpha = 1 / (lambda * t);
    
    % Take stochastic subgradient step
    w = w - alpha * (sg + lambda * w);
    
end

model.w = mean(w_set, 2);
model.predict = @predict;

end

function [yhat] = predict(model, Xhat)
[t,~] = size(Xhat);
Xhat = [ones(t, 1) Xhat];
w = model.w;
yhat = sign(Xhat * w);
end

function [f,sg] = hingeLossSubGrad(w, Xt, y, ~, i)

[d, ~] = size(Xt);

% Function value
wtx = w' * Xt(:, i);
loss = max(0, 1 - y(i) * wtx);
f = loss;

% Subgradient
if loss > 0
    sg = -y(i) * Xt(:, i);
else
    sg = sparse(d, 1);
end

end
