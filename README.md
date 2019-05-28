# Auditing Data Provenance in Text-Generation Models
This repository contains example of experiments for the paper 
Auditing Data Provenance in Text-Generation Models (https://arxiv.org/pdf/1811.00513.pdf/).

### Train text-generation models
The first step is to train target and shadow text-generation models.
To train language model, run the function train_reddit_lm in reddit_lm.py
To train NMT model, run the function train_sated_nmt in sated_nmt.py
To train dialog model, run the function train_cornell_movie in dialogue.py

To train multiple shadow models, set field exp_id=1,2,...n in above function, where n is the number of shadow models. 
Set cross_domain=True to use cross domain datasets for shadow models. An example script for training target model 
and 10 shadow models on reddit language model with 300 users' data:
```python
train_reddit_lm(exp_id=None, cross_domain=False, num_users=300)  # target
for i in range(10):
  train_reddit_lm(exp_id=i, cross_domain=True, num_users=300)  # shadow i
```

### Collect predictied ranks
The next step is to collect predicted ranks on the models we just trained. Use function get_target_ranks and 
get_shadow_ranks to collect the ranks in  reddit_lm_ranks.py, sated_nmt_ranks.py and dialogue_ranks.py. An example 
script for collecting ranks on reddit language model:
```python
get_target_ranks(num_users=300)  # target
for i in range(10):
  get_shadow_ranks(exp_id=i, cross_domain=True, num_users=300)  # shadow i
```

### Auditing user membership
Finally, we can perform auditing on collected ranks. 
Use function user_mi_attack in auditing.py. For example, script for audting reddit language model:
```python
user_mi_attack(data_name='reddit', num_exp=10, num_users=300, cross_domain=True)  # 10 shadow models, 300 users
```

data_name can be 'reddit', 'sated' and 'dialogs'.
