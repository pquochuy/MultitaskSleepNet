clear all
close all
clc

raw_data_path = './raw_data/';
mat_path = './mat/';
if(~exist(mat_path, 'dir'))
    mkdir(mat_path);
end

fs = 100; % sampling frequency
Nsub = 20;

win_size  = 2;
overlap = 1;
nfft = 2^nextpow2(win_size*fs);

for s = 1 : Nsub
    load([raw_data_path, 'n', num2str(s,'%02d'), '.mat']);
    
    eeg_epochs = data(:,:,1);
    % short time fourier transform
    N = size(eeg_epochs, 1);
    X_eeg= zeros(N, 29, nfft/2+1);
    for k = 1 : size(eeg_epochs, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(size(eeg_epochs, 1))]);
        end
        [Xk,~,~] = spectrogram(eeg_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X_eeg(k,:,:) = Xk;
    end
    
    eog_epochs = data(:,:,2);
    % short time fourier transform
    N = size(eog_epochs, 1);
    X_eog= zeros(N, 29, nfft/2+1);
    for k = 1 : size(eog_epochs, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(size(eog_epochs, 1))]);
        end
        [Xk,~,~] = spectrogram(eog_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X_eog(k,:,:) = Xk;
    end
    
    y = single(labels); % one-hot encoding
    
    label = zeros(size(y,1),1);
    for k = 1 : size(y,1)
        label(k) = find(y(k,:) == 1);
    end
    label = single(label);
    
    X = single(X_eeg);
    save([mat_path, 'n', num2str(s,'%02d'), '_cnn_filterbank_eeg.mat'], 'X', 'label', 'y', '-v7.3');
    
    X = single(X_eog);
    save([mat_path, 'n', num2str(s, '%02d'),'_cnn_filterbank_eog.mat'], 'X', 'label', 'y', '-v7.3');
    
end
