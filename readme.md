# On the Relationship Between Interpretation and Contrastive Data Augmentation
## Abstract
Contrastive learning is a popular self-supervised learning method that uses the contrastive views augmented by the unlabeled instance for representation learning. The key issue of contrastive learning is how to construct high-quality contrastive views. Generally speaking, data augmentation is a common method used to generate contrastive views. However, existing methods only randomly sample instances and treat all parts equally, which completely ignores essential semantic information. As a result, it may totally destroy the key information, negatively affecting the quality of the view. To address the above limitations, we leverage interpretation to guide the view generation process, thus, making the augmentation model generate more efficient and reliable views. Since the traditional interpretation methods rely on labels, and there is no class label in comparative learning, we provide novel unsupervised interpretations.
Specifically, based on three representative interpretation methods, we propose their unsupervised versions: unsupervised activation map (UAM), unsupervised gradient-based interpretation (UGI), and unsupervised counterfactual interpretation (UCI). 
The key of these lies in using the data distribution to replace the label information to obtain noteworthy features as unsupervised explanations.
Extensive experiments are built on the benchmark datasets to verify the effectiveness of our method, and compared with the baselines, the improvement is 0.24\% to 1.65\%.

## Environment requirements
* Python (3.8.13)
* Pytorch (1.11.0)
* CUDA
* numpy

## Pretraining
Pretrain the model on CIFAR-10 with ACL(DS)
```bash
python train_simCLR.py ACL_DS --ACL_DS --data /path/to/data
```
Pretrain the model on CIFAR-100 with ACL(DS)
```bash
python train_simCLR.py ACL_DS_CIFAR100 --ACL_DS --dataset cifar100 --data /path/to/data
```
## Finetuning
Adversarial finetune ACL(DS) pretraining model on CIFAR-10 (Need to do ACL(DS) pretraining on CIFAR10 first)
```bash
python train_trades.py ACL_DS_TUNE --checkpoint checkpoints/ACL_DS/model_1000.pt --cvt_state_dict --bnNameCnt 1 --decreasing_lr 40,60 --epochs 100 --data /path/to/data
```
Adversarial finetune ACL(DS) pretraining model on CIFAR-100 (Need to do ACL(DS) pretraining on CIFAR100 first)
```bash
python train_trades.py ACL_DS_CIFAR100_TUNE --dataset cifar100 --checkpoint checkpoints/ACL_DS_CIFAR100/model_1000.pt --cvt_state_dict --bnNameCnt 1 --data /path/to/data
```

# Acknowledge
Trade fine-tuning code from [TRADE](https://github.com/yaodongyu/TRADES) (official code). 
