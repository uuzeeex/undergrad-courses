
% Clear variables and close figures
clear all
close all

% Load data
load basisData.mat % Loads X and y
[n,d] = size(X);

degs = 0 : 10;

% Fit least-squares model
trainError = zeros(1,11);
testError = zeros(1,11);
for deg = 0 : 10
  model = leastSquaresBasis(X,y,deg);

  % Compute training error
  yhat = model.predict(model,X);
  trainError(deg + 1) = sum((yhat - y).^2)/n;
  fprintf('Training error = %.2f\n',trainError(deg + 1));

  % Compute test error
  t = size(Xtest,1);
  yhat = model.predict(model,Xtest);
  testError(deg + 1) = sum((yhat - ytest).^2)/t;
  fprintf('Test error = %.2f\n',testError(deg + 1));
end

plot(0 : 10,smooth(log(trainError)),'r-');
hold on
plot(0 : 10,smooth(log(testError)),'b-');
legend('train', 'test', 'Location', 'northeast')
xlabel('deg')
ylabel('log-error')
  % Plot model
  %figure(1);
  %plot(X,y,'b.');
  %title('Training Data');
%hold on
%Xhat = [min(X):.1:max(X)]'; % Choose points to evaluate the function
%yhat = model.predict(model,Xhat);
%plot(Xhat,yhat,'g');
%xlabel('X')
%ylabel('y')