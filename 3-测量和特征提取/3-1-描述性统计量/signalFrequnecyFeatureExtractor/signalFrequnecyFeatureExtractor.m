% ����һ���򵥵����Ҳ��ź�
Fs = 1000;           % ����Ƶ��
t = 0:1/Fs:1;         % ʱ������
f = 5;               % �ź�Ƶ��
signal = sin(2*pi*f*t); % �ź�
 
% ʹ��signalFrequencyFeatureExtractor��ȡƵ������
[Pxx, f] = signalFrequencyFeatureExtractor(signal, Fs, 'FeatureType', 'POWER SPECTRUM');
 
% ���ƹ�����
plot(f, Pxx);
title('Power Spectrum');
xlabel('Frequency (Hz)');
ylabel('Power');