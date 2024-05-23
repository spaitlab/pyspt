ts = 0:1/fs:0.15;
signal = cos(2*pi*10*ts);
 
subplot(2,1,1)
plot(t,data)
title('Data')
subplot(2,1,2)
plot(ts,signal)
title('Signal')