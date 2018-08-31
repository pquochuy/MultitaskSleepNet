% This script list of training and testing data files which will be
% processed by filterbank-learning DNN networks in tensorflow (for efficiency)


clear all
close all
clc

mat_path = './mat/';

Nfold = 20;
load('./data_split.mat');

tf_path = './tf_data/dnn_filterbank_eeg/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

listing = dir([mat_path, '*_dnn_filterbank_eeg.mat']);
for s = 1 : Nfold   
    disp(['Fold: ', num2str(s),'/',num2str(Nfold)]);
   
	train_s = train_sub{s};
    test_s = test_sub{s};
    
    train_filename = [tf_path, 'train_list_n', num2str(s),'.txt'];
    fid = fopen(train_filename,'wt');
    for i = 1 : numel(train_s)
        sname = listing(train_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    test_filename = [tf_path, 'test_list_n', num2str(s),'.txt'];
    fid = fopen(test_filename,'wt');
    for i = 1 : numel(test_s)
        sname = listing(test_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path

end

tf_path = './tf_data/dnn_filterbank_eog/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end
% list only healthy subjects
listing = dir([mat_path, '*_dnn_filterbank_eog.mat']);
for s = 1 : Nfold   
    disp(['Fold: ', num2str(s),'/',num2str(Nfold)]);
   
	train_s = train_sub{s};
    test_s = test_sub{s};
    
    train_filename = [tf_path, 'train_list_n', num2str(s),'.txt'];
    fid = fopen(train_filename,'wt');
    for i = 1 : numel(train_s)
        sname = listing(train_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    test_filename = [tf_path, 'test_list_n', num2str(s),'.txt'];
    fid = fopen(test_filename,'wt');
    for i = 1 : numel(test_s)
        sname = listing(test_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
end
