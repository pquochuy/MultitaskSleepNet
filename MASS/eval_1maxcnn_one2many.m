% aggregation = 0 % multiplicative aggregation
% aggregation = 1 % additive aggregation
function [acc, kappa, f1, sens, spec, classwise_sens, classwise_sel, C] = eval_1maxcnn_one2many(nfilter, nchan, aggregation)

    addpath('./evaluation/');

    if(nargin == 0)
        nfilter = 1000;
        nchan = 3;
        aggregation = 0;
    end
    
    Ncat = 5;
    
    output_context_size = 3;
    half = floor(output_context_size/2);

    Nfold = 20;
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);

    mat_path = './data_processing/mat/';
	listing = dir([mat_path, '*_cnn_filterbank_eeg.mat']);
	load('./data_processing/data_split_eval.mat');
    
    for fold = 1 : Nfold
        fold
        
        % ground truth
        test_s = test_sub{fold};
        sample_size = zeros(numel(test_s), 1);
        for i = 1 : numel(test_s)
            sname = listing(test_s(i)).name;
            load([mat_path,sname], 'label');
            sample_size(i) = numel(label);
            yt{fold} = [yt{fold}; double(label)];
        end
        
        load(['./tensorflow_net/multitask_1max_cnn_1to3/cnn1d_sleep_357_',num2str(nfilter),'_(08)_eval_',num2str(nchan),'chan/n', num2str(fold), '/test_ret_model_acc.mat']);
        score2 = softmax(score2);
        score1 = softmax(score1);
        score1 = [score1((1+half):end,:); ones(1,Ncat)];
        score3 = softmax(score3);
        score3 = [ones(1,Ncat); score3(1:(end-half),:)];
        if(aggregation == 0)
            score = (score1.* score2 .* score3)/output_context_size;
        else
            score = (score1 + score2 + score3)/output_context_size;
        end
        yhat = zeros(1,size(score,1));
        for i = 1 : size(score,1)
            [~, yhat(i)] = max(score(i,:));
        end
        yh{fold} = double(yhat');
    end
    yh = cell2mat(yh);
    yt = cell2mat(yt);
    
    [acc, kappa, f1, sens, spec] = calculate_overall_metrics(yt, yh);
    [classwise_sens, classwise_sel]  = calculate_classwise_sens_sel(yt, yh);
    C = confusionmat(yt, yh);
end