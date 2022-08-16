# On the Relationship Between Model Interpretation and Contrastive Data Augmentation
## Abstract
Contrastive learning has attracted much attention due to its effectiveness in processing a large amount of unlabeled data. The data augmentation step for generating contrastive views is one of the keys to the success of contrastive learning. Feature perturbation (e.g., cropping or masking) is a widely used data augmentation strategy. Existing methods mainly apply random perturbation on instances, which ignores the meaning of features. As a result, random perturbation could destroy the key information and negatively affect representation learning. To address this limitation, we propose to leverage interpretation to identify the important features which guide the view generation process. 
However, traditional interpretation methods rely on class labels to formulate the objective, while label information is not used in contrastive learning. Meanwhile, interpretation has been diversely defined, and it is not clear how to design the perturbation scheme given different interpretations. To bridge the gaps, we propose unsupervised post-hoc interpretation methods by focusing on explaining representations. Specifically, we design three unsupervised interpretation methods based on three representative supervised methods, including activation map based, gradient-based, and counterfactual interpretation.
Then, we explore how different interpretation methods, combined with different input perturbation strategies, affect the effectiveness of contrastive learning.
Extensive experiments on the benchmark datasets verify the effectiveness of our method compared with baseline methods.

## Environment requirements
* Python (3.8.13)
* Pytorch (1.11.0)
* CUDA
* numpy



# Acknowledge
Trade fine-tuning code from [TRADE](https://github.com/yaodongyu/TRADES) (official code). 
