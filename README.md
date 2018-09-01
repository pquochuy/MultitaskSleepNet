# MultitaskSleepNet
- Huy Phan, Fernando Andreotti, Navin Cooray, Oliver Y. Chén, and Maarten De Vos. [__Joint Classification and Prediction CNN Framework for Automatic Sleep Stage Classification.__](https://arxiv.org/pdf/1805.06546) _arXiv Preprint arXiv:1805.06546,_ 2018

These are source code and experimental setup for two sleep databases: __SleepEDF Expanded database__ and __MASS database__, used in our above arXiv preprint. Although the networks have many things in common, we try to separate them and to make them work independently to ease exploring them invididually.

You need to download the databases to run the experiments again
- __SleepEDF Expanded database__ can be downloaded from [here](https://www.physionet.org/pn4/sleep-edfx/). We also included a Matlab script that you can use for downloading.
- __MASS database__ is available here [here](https://massdb.herokuapp.com/en/). Information on how to obtain it can be found therein.

Currently for __MASS database__, _Tsinalis et al._'s network and _DeepSleepNet1_ (_Supratak et al._) are still missing. We are currently cleaning them up and and will update them very shortly.

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

