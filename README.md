1. prepare torch code

2. modify it to get random sized image

3. test it on TPU (XLA)


================================================

Phase 1. 
Once-for-all 에서 하면 좋음

depth, width, resol ? 
1. (d,w,r) 선택
```python
d = np.random.randint()
w = 
r = 
```

2. inference -> acc, latency
```python
from ofa.model_zoo import ofa_net
# Manually set the sub-network
ofa_network.set_active_subnet(ks=7, e=6, d=4)
manual_subnet = ofa_network.get_active_subnet(preserve_weight=True)
```
```bash
python eval_ofa_net.py --path 'Your path to imagenet' --net ofa_mbv3_d234_e346_k357_w1.0
```

```bash
./save_network_acc_latench.py
```
[(d,w,r) * n] -> network

3. dataset은 cifar

</br> 이 샘플 한 1만개 뽑아보기

 ## 구체화
 1. Input Pipeline 구현
    - run_manager modification
 2. Training Logic 구현
    - cifar-100
    - cosine annhealing
    - BSCONV 의 training logic 사용해서
 ```python
image = resize(image)
model = subnet
```
 
 3시 ~ 5시 sync 시간
 
 * 재성 : model 구현
    - get_active_subnet(d,w,r) 구현
    - 
================================================
## Issues
- where's the data on ofa?

- pytest 시 path 가 좀 다른가? ofa 를 못알아먹네





