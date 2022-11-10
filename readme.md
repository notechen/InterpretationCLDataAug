# Interpretation-Guided Data Augmentation for Contrastive Learning
## Abstract
Contrastive learning has attracted much attention due to its effectiveness in processing unlabeled data. Data augmentation (e.g., image cropping or masking) for generating contrastive views is one of the keys to the success of contrastive learning. However, existing augmentation mainly applies random perturbation on instances, which ignores the semantics of image features. Random perturbation could fail to preserve key information or generate discriminative views, thus negatively affecting the learning process. 
To address this limitation, we propose to leverage interpretation to identify important and discriminative features for guiding the generation of contrastive views. 
Nevertheless, traditional interpretation methods rely on class labels and trained models, which are not available during contrastive learning. Meanwhile, there are multiple definitions for interpretation, where it is not clear how to design the augmentation scheme given different interpretations. To bridge the gaps, we propose unsupervised post-hoc interpretation for explaining latent representations. 
Then, we explore how different image augmentation schemes, combined with different interpretation methods, affect the effectiveness of contrastive learning. Extensive experiments on the benchmark datasets verify the effectiveness of our method compared with baseline methods.

## Environment requirements
* Python (3.8.13)
* Pytorch (1.11.0)
* CUDA
* numpy



# Acknowledge
Trade fine-tuning code from [TRADE](https://github.com/yaodongyu/TRADES) (official code). 
