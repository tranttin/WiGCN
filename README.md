# Weighted addition input matrix GCN (WiGCN)

Two main contributions
- Calculating the additional matrix
- New model WiGCN.

This is our improvement Tensorflow based on code of https://github.com/xiangwang1223/neural_graph_collaborative_filtering


## Introduction
In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation. We propose a new model named LightGCN, including only the most essential component in GCN—neighborhood aggregation—for collaborative filtering.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.11.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1
* cython == 0.29.15

## Examples to run a 3-layer WiGCN

### Gowalla dataset
* Command
```
python WiGCN.py --dataset gowalla --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000
```

### Yelp2018 dataset
* Command
```
python WiGCN.py --dataset yelp2018 --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000
```

### Amazon-book dataset
* Command
```
python WiGCN.py --dataset amazon-book --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 8192 --epoch 1000
```
* Output log :
```

NOTE : the duration of training and testing depends on the running environment.
## Dataset
We provide three processed datasets: Gowalla, Yelp2018 and Amazon-book.
* `train.txt`
  * Train file.
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.

* `test.txt`
  * Test file (positive instances).
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.
  
* `user_list.txt`
  * User file.
  * Each line is a triplet (org_id, remap_id) for one user, where org_id and remap_id represent the ID of the user in the original and our datasets, respectively.
  
* `item_list.txt`
  * Item file.
  * Each line is a triplet (org_id, remap_id) for one item, where org_id and remap_id represent the ID of the item in the original and our datasets, respectively.


=======
