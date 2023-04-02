# ML2023_Spring ReadMe

## How to reproduce my results

#### Download Datasets
```
pip install gdown --upgrade
gdown --id '1tbGNwk1yGoCBdu4Gi_Cia7EJ9OhubYD9' --output food11.zip
apt-get install zip
unzip food11.zip
```
#### Run codes
Run these 4 python files respectively and waiting for 4 predicted .CSV files.
```
python model_efficientnetV2.py
python model_resnext101.py
python model_SHUFFLENET.py
python model_vgg19bn.py
```

#### Ensemble my CSV File
```
python ensemble.py
```
I take each Kaggle Public Scores as the corresponding voting weight and average them to get the final file after ensembling. So if you get a slightly diffident Kaggle score when retraining, it is recommended that you change the weight list in the `ensemble.py` file to avoid affecting the final result.

#### Show the t-SNE figure
```
python tSNE.py
```
In `tSNE.py`, I used VGG_19_bn to generate the image. To use another model just change the index in the text.

## Others*

GPU: I use a V100-32G, so CUDA out-of-memory may occur during retraining.

CPU: I am using an excellent CPU in TWCC, num_worker is 8 or even 16 in some DataLoaders, so people need to adjust it to 4 or 0 during re-produce. 

Training time: VGG19BN takes 10 hours to train, ResNext101 takes 22 hours to train, and other models take about 12 hours on average to train.
