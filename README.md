# MODEL AND DATASET OF SESSION-BASED RECOMMENDATION PROBLEM

## DATASETS
- Yoochoose: Dataset of Recsys 2015 Challenge.
More information and download in: https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015/data
- Updating...

## MODEL
- NARM: A model using GRU
- SR-GNN: First model using GNN 
- GCSAN: SR-GNN + Attention :D
- TAGNN++: Seem like GCSAN + target attention (seem super slow training time)
- Updating....

## Run code:
### Training:
Run train.py in the folder having the name of the model to train the model. 
Time training using GPU in time.txt file

### Testing:
Run test.py in the folder having the name of the model
```shell
python SRGNN/test.py
```