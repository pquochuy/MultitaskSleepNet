clear all
close all
clc


addpath('./sleepedfx_scripts/');
addpath('./edf_reader/');

%% you can comment out these lines to download the data SleepEDFx data from Physionet
%download_dir = './sleepedfx_database/';
%[saved_file, status] = my_downloadEDFxData(download_dir); % download data
%% the annotation url is not availabel anymore, so I copied them here
%% So you do not need to download them again
%%my_downloadEDFxAnnotations(download_dir); % download annotation

data_path = './sleepedfx_database/';

raw_data_path = './raw_data/';
if(~exist(raw_data_path, 'dir'))
    mkdir(raw_data_path);
end

% Sampling rate of hypnogram (default is 30s)
epoch_time = 30;
fs = 100; % sampling frequency

Nsub = 20; % numer of subjects

data_all = cell(Nsub,1);
label_all = cell(Nsub,1);
y_all = cell(Nsub,1); % one-hot encoding

% list only healthy (SC) subjects
listing = dir([data_path, 'SC4*']);

for i = 1 : numel(listing)
    disp(listing(i).name)
    target_dir = [data_path, listing(i).name, '/'];
    
    [~,filename,~] = fileparts(listing(i).name);
    [sub_id,~] = edfx_dir2sub(filename);
    sub_id = sub_id + 1; % index 0 to 1
    
    % load edf data to get Fpz-Cz, and EOGhorizontal channels
    edf_file = [target_dir, listing(i).name, '-PSG.edf'];
    [header, edf] = edfreadUntilDone(edf_file);
    channel_names = header.label;
    
    for c = 1 : numel(channel_names)
        channel_names{c} = strtrim(channel_names{c});
    end
    chan_ind_eeg = find(ismember(channel_names, 'EEGFpzCz'));
    if(isempty(chan_ind_eeg))
        disp('Oops, wait. Channel not found!');
        pause;
    end
    
    if(header.frequency(chan_ind_eeg) ~= fs)
        disp('Oops, wait! Sampling frequency mismatched!');
        pause;
    end
    
    chan_ind_eog = find(ismember(channel_names, 'EOGhorizontal'));
    if(isempty(chan_ind_eog))
        disp('Oops, wait. Channel not found!');
        pause;
    end
    
    if(header.frequency(chan_ind_eog) ~= fs)
        disp('Oops, wait! Sampling frequency mismatched!');
        pause;
    end
    
    chan_data_eeg = edf(chan_ind_eeg, :);
    chan_data_eog = edf(chan_ind_eog, :);
    clear edf header channel_names
    
    % ensure the signal is calibrated to microvolts
    if(max(chan_data_eeg) <= 10)
        disp('Signal calibrated!');
        chan_data_eeg = chan_data_eeg * 1000;
    end
    % zero-mean
    chan_data_eeg = chan_data_eeg - mean(chan_data_eeg);
    % zero-mean
    chan_data_eog = chan_data_eog - mean(chan_data_eog);
    
    % load hypnogram
    hyp_file = fullfile([target_dir, 'info/', listing(i).name, '.txt']);
    hypnogram = edfx_load_hypnogram( hyp_file );
    
    % process times to determine the in-bed duration
    [chan_data_eeg_new, chan_data_eog_new, hypnogram_new] = edfx_process_time_2chan(target_dir, chan_data_eeg, chan_data_eog, hypnogram, epoch_time, fs);
    
    eeg_epochs = buffer(chan_data_eeg_new, epoch_time*fs);
    eeg_epochs = eeg_epochs';
    
    eog_epochs = buffer(chan_data_eog_new, epoch_time*fs);
    eog_epochs = eog_epochs';

    label = edfx_hypnogram2label(hypnogram_new);
    % excluding Unknown and non-score
    ind = (label == 0);
    disp([num2str(sum(ind)), ' epochs excluded.'])
    label(ind) = [];
    eeg_epochs(ind,:,:) = [];
    eog_epochs(ind,:,:) = [];

    data = zeros([size(eeg_epochs), 2]);
    data(:,:,1) = eeg_epochs;
    data(:,:,2) = eog_epochs;
    
    % since the d
    y = zeros(numel(label),1);
    for k = 1 : numel(label)
        y(k, label(k)) = 1;
    end
    
    if(isempty(data_all{sub_id}))
        data_all{sub_id} = data;
        y_all{sub_id} = y;
    else
        data_all{sub_id} = cat(1, data_all{sub_id}, data);
        y_all{sub_id} = [y_all{sub_id}; y];
    end
    clear X_eeg X_eog label y
end

patinfo.ch_orig{1} = "FpzCz";
patinfo.ch_orig{1} = "EOGhorizontal";
patinfo.fs = 100;
patinfo.chlabels = {"EEG", "EOG"};
patinfo.classes = {"W", "R", "N1", "N2", "N3"};

for s = 1 : Nsub
    labels = logical(y_all{s});
    data = data_all{s};
    save([raw_data_path, 'n', num2str(s,'%02d'), '.mat'], 'data', 'labels', 'patinfo', '-v7.3');
end


