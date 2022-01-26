# WSDM 2022 Temporal Link Prediction Challenge-We can [mask]!

[WSDM Cup Website link](https://www.wsdm-conference.org/2022/call-for-wsdm-cup-proposals/)

[Link to this challenge](https://www.dgl.ai/WSDM2022-Challenge/)

## Reproduction steps 

- Create the environment

```
conda create -n WSDM python=3.7
```

- Activate the environment

```
conda activate WSDM
```

- Install required packages

```
pip install -r requirements.txt
```

Note: GPU is required for faster training (8 GB Memory at least).  

PS: If you face the problem of "Could not load dynamic library xxx for Tensorflow GPU", the Answer 2 in https://www.mashen.zone/thread-3634622.htm?user=4 is suggested.

- Dataset location

**finals** contains  the final test set

**intermediate** contains the updated  [input_A_initial.csv.gz](https://data.dgl.ai/dataset/WSDMCup2022/input_A_initial.csv.gz) and [input_B_initial.csv.gz](https://data.dgl.ai/dataset/WSDMCup2022/input_B_initial.csv.gz) files, as well as the intermediate test set [input_A.csv.gz](https://data.dgl.ai/dataset/WSDMCup2022/intermediate/input_A.csv.gz) and [input_B.csv.gz](https://data.dgl.ai/dataset/WSDMCup2022/intermediate/input_B.csv.gz)

**train_csvs** contains the training data

```
│  main.py
│  model.py
│  README.md
│  requirements.txt
│  utlis.py
│          
├─finals
│      input_A_final.csv
│      input_B.csv
│      
├─intermediate
│      input_A.csv
│      input_A_initial2.csv
│      input_B.csv
│      input_B_initial2.csv
│      
├─train_csvs
       edges_train_A.csv
       edges_train_B.csv
       edge_type_features.csv
       node_features_sampled.csv
```

- Usage

```
# for dataset A
python main.py --dataset A --epochs 9 --emb_dim 200 --negative_samples 5
# for dataset B
python main.py --dataset B --epochs 12 --emb_dim 100 --negative_samples 1
```

PS: the total running time for two dataset is less 2 hours using NVIDIA GeForce RTX 3070 Ti.

- Result

```
Intermediate Test Set
- output_A_inter.csv
- output_B_inter.csv
Final Test Set
- output_A.csv
- output_B.csv
```

- The result of Intermediate Test Set

```
AUC of Dataset A: 0.604264
AUC of Dataset B: 0.898969
Overall result: 0.722729
```

- The result of Intermediate Test Set

```
AUC of Dataset A: 0.603621
AUC of Dataset B: 0.898232
```
