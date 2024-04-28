clear ans
close all

x = [0,0];
u = 0;
F = @(x,u) (x(1))/(pow2(x(1)) + 1) + power(u(1),3)/(pow2(x(2)) + 1) - u;
A = 1;
B = 1;
t = 1000;

F_output = zeros(t,1);
input = zeros(t,1);

for i = 1:t
    temp = x(1);
    x(1) = F(x,u);
    x(2) = temp;
    
    input(i) = u;
    F_output(i) = x(1);

     u = A*sin(i*pi/50) + B*sin(i*pi/20) ;% tutaj wersja z pewnym porandomizowanym wejściem A*sin(i*pi/50)*rand() + B*sin(i*pi/20)*rand();
end

figure(1);
hold on;
plot(1:t,F_output);
figure(2);
plot(1:t,input);