# TTEN
This is the official implementation of Test Time Embedding Normalization for Popularity Bias Mitigation, CIKM 2023.

## Requirements
- python == 3.9.12
- pytorch == 1.13.0
- scipy == 1.11.1
- numpy == 1.25.2
- pandas == 2.0.3
- tqdm == 4.66.1
- scikit-learn == 1.3.0

Clone the repository and install requirements with
```
conda create -n TTEN python=3.9.12
conda activate TTEN

pip install -r requirements.txt
```

## Run the Code
### Gowalla 
`python main.py --loss_type ssm --lr 0.001 --ssm_temp 0.1 --dataset fair_gowalla --tten`

### Yelp2018 
`python main.py --loss_type ssm --lr 0.001 --ssm_temp 0.12 --dataset fair_yelp2018 --tten`

### ML10M 
`python main.py --loss_type ssm --lr 0.001 --ssm_temp 0.1 --dataset fair_ml10m --tten`

## Citation
```
   
```