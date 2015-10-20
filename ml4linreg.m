- Load the data
x = load('ml4Linx.dat');
y = load('ml4Liny.dat');
- Plot
plot(x, y, 'o');
xlabel('x');
ylabel('y');
hold on;
[m,n] = size(x);
 
- 5th order polynomial
x = [ones(m,1), x, x.^2, x.^3, x.^4, x.^5];

- Initialize theta
theta = zeros(size(x(1,:)))';

- Set lambda (0, 1, 10)
lambda = 0;
L = lambda .* eye(6);
L(1) = 0;
theta = (x' * x + L)\(x' * y);

- L2 norm
theta_norm = norm(theta);

- Set x axis
x_vals = (-1:0.05:1)';

- Set regularization line
feats = [ones(size(x_vals)), x_vals, x_vals .^2,...x_vals.^3, x_vals.^4, x_vals.^5];

- Plot 
plot(x_vals, feats * theta, '--');
xlabel('x_vals');
ylabel('features');
legend('Training data', 'Lambda = 0');
