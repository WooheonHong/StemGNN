# Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting

This repository is the wooheon's reproducing of StemGNN



## Requirements


```setup
pip install --upgrade pip
pip install -r requirements.txt
```

## Datasets

https://github.com/microsoft/StemGNN

## Training and Evaluation


```train & evaluate
python main.py --dataset <name of csv file> --window_size <length of sliding window> --horizon <predict horizon> --norm_method z_score --batch_size 64 --train_length 7 --validate_length 2 --test_length 1
```

The detailed descriptions about the parameters are as following:
New parameter are bold type.

| Parameter name | Description of parameter |
| --- | --- |
| train | whether to enable training, default True |
| evaluate | whether to enable evaluation, default True |
| dataset | file name of input csv |
| window_size | length of sliding window, default 12 |
| horizon | predict horizon, default 3 |
| train_length | length of training data, default 7 |
| validate_length | length of validation data, default 2 |
| test_length | length of testing data, default 1 |
| epoch | epoch size during training |
| lr | learning rate |
| **attention_layer** | hyper parameter of STemGNN which controls the parameter number of attention layers, default 32|
| **randomwalk_laplacian** | determine whether to use randomwalk normalized laplacian matrix|
| multi_layer | hyper parameter of STemGNN which controls the parameter number of hidden layers, default 5 |
| device | device that the code works on, 'cpu' or 'cuda:x' | 
| validate_freq | frequency of validation |
| batch_size | batch size, default 64 |
| norm_method | method for normalization, 'z_score' or 'min_max' |
| early_stop | whether to enable early stop, default False |

## Results

My reproducing model shows following performance on the 10 datasets:

**Table 1** Configuration and perforamance for all datasets
| Dataset | window_size | horizon | norm_method | MAE  | RMSE | MAPE(%) |
| -----   |--- |---- | --- |---- | ---- | ---- |
| METR-LA | 12 | 3 | z_score |
| PEMS-BAY | 12 | 3 | z_score |
| PEMS03 | 12 | 3 | z_score |
| PEMS04 | 12 | 3 | z_score |
| PEMS07 |12 | 3 | z_score |
| PEMS08 | 12 | 3 | z_score |
| COVID-19 | 28 | 28 | z_score |

