% This script prepares data into a format which is ready to train a filterbank-learning DNN on Tensorflow
clear all
close all
clc

mat_path = './mat/';

%% EEG
listing = dir([mat_path, '*_cnn_filterbank_eeg.mat']);
Nsub = numel(listing);
for s = 1 : Nsub
    disp(['Subject: ', num2str(s),'/',num2str(Nsub)]);
    
    load([mat_path, listing(s).name]);

    [Ntrain,T,F,Nchan] = size(X);
    X_all = X;
    y_all = y;
    label_all = label;
    clear X y label
        
    X_t = cell(Ntrain, 1);
    label_t = cell(Ntrain,1);
    y_t = cell(Ntrain,1);
    for i = 1 : Ntrain
        X_t{i} = squeeze(X_all(i, :, :,1));
        label_t{i} = ones(T, 1)*label_all(i);
        y_t{i} = repmat(y_all(i,:),T,1);
    end
    X = cell2mat(X_t);
    label = cell2mat(label_t);
    y = cell2mat(y_t);
    clear X_t label_t y_t
    sname = strrep(listing(s).name,'cnn_filterbank_eeg','dnn_filterbank_eeg');
    save([mat_path, sname], 'X', 'label', 'y', '-v7.3');
    clear X label y
end

%% EOG
listing = dir([mat_path, '*_cnn_filterbank_eog.mat']);
Nsub = numel(listing);
for s = 1 : Nsub
    disp(['Subject: ', num2str(s),'/',num2str(Nsub)]);
    
    load([mat_path, listing(s).name]);

    [Ntrain,T,F,Nchan] = size(X);
    X_all = X;
    y_all = y;
    label_all = label;
    clear X y label
        
    X_t = cell(Ntrain, 1);
    label_t = cell(Ntrain,1);
    y_t = cell(Ntrain,1);
    for i = 1 : Ntrain
        X_t{i} = squeeze(X_all(i, :, :,1));
        label_t{i} = ones(T, 1)*label_all(i);
        y_t{i} = repmat(y_all(i,:),T,1);
    end
    X = cell2mat(X_t);
    label = cell2mat(label_t);
    y = cell2mat(y_t);
    clear X_t label_t y_t
    sname = strrep(listing(s).name,'cnn_filterbank_eog','dnn_filterbank_eog');
    save([mat_path, sname], 'X', 'label', 'y', '-v7.3');
    clear X label y
end

%% EMG
listing = dir([mat_path, '*_cnn_filterbank_emg.mat']);
Nsub = numel(listing);
for s = 1 : Nsub
    disp(['Subject: ', num2str(s),'/',num2str(Nsub)]);
    
    load([mat_path, listing(s).name]);

    [Ntrain,T,F,Nchan] = size(X);
    X_all = X;
    y_all = y;
    label_all = label;
    clear X y label
        
    X_t = cell(Ntrain, 1);
    label_t = cell(Ntrain,1);
    y_t = cell(Ntrain,1);
    for i = 1 : Ntrain
        X_t{i} = squeeze(X_all(i, :, :,1));
        label_t{i} = ones(T, 1)*label_all(i);
        y_t{i} = repmat(y_all(i,:),T,1);
    end
    X = cell2mat(X_t);
    label = cell2mat(label_t);
    y = cell2mat(y_t);
    clear X_t label_t y_t
    sname = strrep(listing(s).name,'cnn_filterbank_emg','dnn_filterbank_emg');
    save([mat_path, sname], 'X', 'label', 'y', '-v7.3');
    clear X label y
end

