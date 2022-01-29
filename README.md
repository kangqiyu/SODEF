# Stable Neural ODE with Lyapunov-Stable Equilibrium Points for Defending Against Adversarial Attacks
  
[Stable Neural ODE with Lyapunov-Stable Equilibrium Points for Defending Against Adversarial Attackss](https://openreview.net/forum?id=9CPc4EIr2t1).

Qiyu Kang, Yang Song, Qinxu Ding, Wee Peng Tay

## Environment settings

- OS: Ubuntu 18.04
- GPU: RTX 2080 Ti, RTX a5000, RTX 3090
- Cuda: 11.1 or 10.2
- Python: >=3.6
- PyTorch: >= 1.6.0
- Torchvision: >= 0.7.0

## Empirical Evaluations

### Compatibility of SODEF

#### In this section, we show compatibility of SODEF using [TRADES](https://github.com/P2333/Bag-of-Tricks-for-AT/):

We append our SODEF after TRADES net to improve the model robustness against adversarial attacks. TRADES works as the feature extractor <img src="https://render.githubusercontent.com/render/math?math=h_{\boldsymbol{\phi}}"> as in our paper. Please note TRADES weights are kept fixed during the training. We use the pretrained model provided by [TRADES Repo](https://github.com/P2333/Bag-of-Tricks-for-AT/).

<span id="tab:r2_3" label="tab:r2_3"></span>

<div id="tab:r2_3">

|         Attack / Model         | TRADES ℒ<sub>∞</sub> | TRADES+SODEF ℒ<sub>∞</sub> | TRADES ℒ<sub>2</sub> | TRADES+SODEF ℒ<sub>2</sub> |
|:------------------------------:|:--------------------:|:--------------------------:|:--------------------:|:--------------------------:|
|             Clean              |        85.48         |           85.18            |        85.48         |           85.18            |
|       APGD<sub>CE</sub>        |        56.08         |           __70.90__            |        61.74         |           __74.35__            |
| APGD<sub>DLR</sub><sup>T</sup> |        53.70         |           __64.15__            |        59.22         |           __68.55__            |
|        FAB<sup>T</sup>         |        54.18         |           __82.92__            |        60.31         |           __83.15__            |
|             Square             |        59.12         |           __62.21__            |        72.65         |           __76.02__            |
|           AutoAttack           |        53.69         |           __57.76__            |        59.42         |           __67.75__            |

Tab 1. Classification accuracy (%) using TRADES (w/ and w/o SODEF) under
[AutoAttack](https://github.com/fra31/auto-attack) on adversarial CIFAR10 examples with ℒ<sub>2</sub> norm
(*ϵ* = 0.5) and ℒ<sub>∞</sub> norm (*ϵ* = 8/255).

</div>

__Transfer attack__:

Classification accuracy for adv examples generated from original [pretrained model](https://github.com/P2333/Bag-of-Tricks-for-AT/) using AA ℒ<sub>∞</sub> (*ϵ* = 8/255) attacks : 61.94%.

```python
cd trades_r
python sodef_eval_ode.py
```
```python
cd trades_r
sodef_eval_transfer.ipynb
```

#### In this section, we show compatibility of SODEF using [Rebuffi2021](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness):


Similar to the above section, we append SODEF after the pretrained model provided by the [RobustBench](https://github.com/RobustBench/robustbench) with keywords "Rebuffi2021Fixing_70_16_cutmix_extra". The weights are kept fixed during the training except the final FC layer. The pretrained model without SODEF achieves 66.58% accuracy under AutoAttack. We show that with SODEF, the robust accuracy could be improved to over 70%:


<span id="tab:r2_3" label="tab:r2_3"></span>

<div id="tab:r2_3">

| Attack / Model |   Rebuffi2021    | Rebuffi2021 +SODEF | Transfer Attack |
|:--------------:|:-----:|:------:|:---------------:|
|     Clean      | 92.23 |   93.73    |       NA        |  
|   AutoAttack   | 66.58 |   __71.28__    |       ?      | 

Tab 2. Classification accuracy (%) using (w/ and w/o SODEF) under AutoAttack on
adversarial CIFAR10 examples with ℒ<sub>∞</sub> norm (*ϵ* = 8/255).

</div>

Here again for the transfer attack, adv examples are generated from original pretrained model using AA ℒ<sub>∞</sub> (*ϵ* = 8/255) attacks.


```python
cd Rebuffi2021Fixing_70_16_cutmix_extra
python sodef_eval_ode.py
```
```python
cd Rebuffi2021Fixing_70_16_cutmix_extra
sodef_eval_transfer.ipynb
```

### Notification
More test code and models will be uploaded soon after packing.

We currenly only upload the test code for SODEF. Please understand we have strict protocols for code release as this research is partially funded by corporate funding. We will upload the training code as soon as permission is granted.
