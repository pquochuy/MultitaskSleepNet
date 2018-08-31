function [acc, kappa, f1, sens, spec, classwise_sens, classwise_sel, C] = eval_deepcnn_one2one(nchan)

    if(nargin == 0)
        nchan = 2;
    end
    
    Ncat = 5;

    Nfold = 20;
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);
     for fold = 1 : Nfold
         fold
         load(['./tensorflow_net/deep_cnn_baseline_1to1/deepcnn_sleep_96_96_1024_1024_(08)_eval_',num2str(nchan),'chan/n', num2str(fold), '/test_ret_model_acc.mat']);
         yh{fold} = double(yhat');

         % load ground-truth labels
        load(['./data_processing/mat/n', num2str(fold,'%02d'), '_cnn_filterbank_eeg.mat'], 'label');
        yt{fold} = double(label);
     end
    yh = cell2mat(yh);
    yt = cell2mat(yt);
    
    [acc, kappa, f1, sens, spec] = calculate_overall_metrics(yt, yh);
    [classwise_sens, classwise_sel]  = calculate_classwise_sens_sel(yt, yh);
    C = confusionmat(yt, yh);
end