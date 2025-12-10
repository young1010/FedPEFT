## Datasets
- Public datasets used in paper are from Amazon[https://nijianmo.github.io/amazon/index.html] and MovieLens-1M[https://grouplens.org/datasets/movielens/1m/]
- Use and set datasets in config/data_config.yaml

## Model
- Set models config in config/model_config.yaml
- See models in zoo/

## Run
- Run a model through:

```bash
python main.py --expid {expid in model_config.yaml} --gpu {-1 for cpu, others for cuda}
```

for example:

```bash
python main.py --expid FedMF_ml1m_triple_101 --gpu 0
```

## Requirements
* PyYAML
* pandas
* scikit-learn
* numpy
* h5py
* tqdm
See more in requirements.txt
  
## Acknowledgement
- The framework of this code is based on Fuxictr[https://github.com/reczoo/FuxiCTR]