Link to paper https://arxiv.org/abs/1812.03282.
Link to dataset https://www.kaggle.com/pengcw1/market-1501/data

Please change the path to datasets where ever required.

## Market1501
### Data prepare
``` 
python3 prepare.py 
```
### Train (appearance feature learning)
```
python3 train_market.py 
```

### Test (appearance feature extraction)
```
python3 test_st_market.py 
```
### Generate st model (spatial-temporal distribution)
```
python3 gen_st_model_market.py 
```
### Evaluate (joint metric, you can use your own visual feature or spatial-temporal streams)
```
python3 evaluate_st.py 
```

The code is mainly taken from this repository https://github.com/Wanggcong/Spatial-Temporal-Re-identification, all the credits go to the repository contributors.

## Citation
```
@article{guangcong2019aaai,
  title={Spatial-Temporal Person Re-identification},
  author={Wang, Guangcong and Lai, Jianhuang and Huang, Peigen and Xie, Xiaohua},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={8933-8940},
  year={2019}
}
```
