- Load and plot positive and negative class
x = load('ml4Logx.dat');
y = load('ml4Logy.dat');
pos = find(y == 1);
neg = find(y == 0);
plot(x(pos,1), x(pos,2), '+');
hold on;
plot(x(neg,1), x(neg,2), 'o');
xlabel('Feature 1');
ylabel('Feature 2');
legend('y = 1', 'y = 0');

- Map feature
x = map_feature(x(:,1),x(:,2));
[m,n] = size(x);

- Initialize theta
theta = zeros(n, 1);

- Sigmoid
g = inline('1.0 / (1.0 + exp(-z))');

- Initialize J and iterations
num = 15;
J = zeros(num, 1);

- Set lambda (0,1,10)
lambda = 0;

- Newton’s method
for i = 1:num
    z = x * theta;
    h = g(z);
    
    J(num) =(1/m)*sum(-y.*log(h) - (1-y).*log(1-h)) +...
        (lambda/(2*m))*norm(theta([2:end]))^2; 
    G = (lambda/m).*theta;
    G(1) = 0;
    L = (lambda/m).*eye(n);
    L(1) = 0;
    grad = ((1/m).*x' * (h-y)) + G;
    H = ((1/m).*x' * diag(h) * diag(1-h) * x) + L;
    theta = theta - H\grad;
end

- Print J for each lambda
figure
plot(0:MAX_ITR-1, J, 'o');
xlabel('Iteration'); 
ylabel('J');

- Decision boundary line
u = linspace(-1, 1.5, 200);
v = linspace(-1, 1.5, 200);
z = zeros(length(u), length(v));
for i = 1:length(u)
    for j = 1:length(v)
        z(i,j) = map_feature(u(i), v(j))*theta;
    end
end
z = z';
contour(u, v, z, [0, 0], 'LineWidth', 2)
legend('y = 1', 'y = 0', 'Decision boundary')
title(sprintf('\\lambda = %g', lambda), 'FontSize', 14)
hold off
