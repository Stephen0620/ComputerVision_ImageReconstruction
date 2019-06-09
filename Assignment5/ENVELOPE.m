t = 0:1/1000:3;
q1 = chirp(t,30,2,5).*exp(-(2*t-3).^2)+2;
figure;
plot(t,q1)

[up,~] = envelope(q1,300);

hold on
plot(t,up,'linewidth',1.5)
legend('q1','up')
ylim([1,3]);
hold off