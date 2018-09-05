# MultitaskSleepNet
- Huy Phan, Fernando Andreotti, Navin Cooray, Oliver Y. Chén, and Maarten De Vos. [__Joint Classification and Prediction CNN Framework for Automatic Sleep Stage Classification.__](https://arxiv.org/pdf/1805.06546) _arXiv Preprint arXiv:1805.06546,_ 2018

These are source code and experimental setup for two sleep databases: __SleepEDF Expanded database__ and __MASS database__, used in our above arXiv preprint. Although the networks have many things in common, we try to separate them and to make them work independently to ease exploring them invididually.

You need to download the databases to run the experiments again
- __SleepEDF Expanded database__ can be downloaded from [here](https://www.physionet.org/pn4/sleep-edfx/). We also included a Matlab script that you can use for downloading.
- __MASS database__ is available here [here](https://massdb.herokuapp.com/en/). Information on how to obtain it can be found therein.

Currently for __MASS database__, _Tsinalis et al._'s network and _DeepSleepNet1_ (_Supratak et al._) are still missing. We are currently cleaning them up and and will update them very shortly.

How to run:
-------------
1. Download the databases
2. Data preparation
- Change directory to _\[database\]/data_processing_, for example _MASS/data_processing_
- Run _main_run.m_
3. Network training and testing
- Change directory to a specific network in _\[database\]/tensorflow_net_, for example _MASS/tensorflow_net/multitask_1max_cnn_1to3/_
- Run a bash script, for example _bash run_3chan.sh_ to repeat 20 cross-validation folds.  
_Note1:_ You may want to modify and script to make use of your computational resources, such as place a few process them on multiple GPUs. If you want to run multiple processes on a single GPU, you may want to modify the Tensorflow source code to change __GPU options__ when initializing a Tensorflow session.  
_Note2:_ All networks, except those based on raw signal input like _Chambon et al._, _DeepSleepNet1_ (_Supratak et al._), _Tsinalis et al._ on MASS database, require pretrained filterbanks for preprocessing. If you want to repeat everything, you may want to train the filterbanks first by executing the bash script in _\[database\]/tensorflow_net/dnn-filterbank/_
4. Evaluation
- Go up to _\[database\]/_ directory, for example _MASS/_
- Execute a specific evaluation Matlab script, for example _eval_1maxcnn_one2many.m_

Environment:
-------------
- Matlab v7.3 (for data preparation)
- Python3
- Tensorflow GPU 1.3.0 (for network training and evaluation)

Contact:
-------------
Huy Phan  
Institute of Biomedical Engineering  
Department of Engineering Science  
University of Oxford  
Email: huy.phan{at}eng.ox.ac.uk

