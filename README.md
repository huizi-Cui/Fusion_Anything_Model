# Generalized Multimodal Foundation Model

Huizi Cui<sup>1,†</sup>, Zongbo Han<sup>2,†,‡</sup>, Chenggong Ding<sup>1</sup>, Naichuan Xiao<sup>1</sup>, Jialong Yang<sup>1</sup>, Jingdong Chen<sup>1</sup>, Guangyu Wang<sup>2</sup>, Qinghua Hu<sup>1</sup>, Changqing Zhang<sup>1,*</sup>

<sup>1</sup> Tianjin University <sup>2</sup> Beijing University of Posts and Telecommunications

<sup>†</sup> Equal contribution <sup>‡</sup> Project lead <sup>*</sup> Corresponding author

📄 **Paper:** [PDF](./paper.pdf)

## Overview
Making prediction with multimodal data is widely used in diverse scenarios. Existing multimodal fusion models, once deployed, can only handle predefined modalities (e.g., vision, text and audio) and single tasks, making it difficult to quickly adapt to new downstream applications. Therefore, a natural yet rather aggressive question arises, whether there exists a general multimodal fusion model that can be applied to arbitrary modality combinations and arbitrary prediction tasks. We argue that a unified multimodal fusion model should not depend on specific modalities and instead encode transferable patterns of multimodal correlation. To this end, we propose a simple and effective learning paradigm based on training over the generation of large-scale synthetic multimodal datasets with diverse causal structures that formally characterize the generative processes of multimodal data in real world. Building on this framework, we propose the generalized multimodal foundation model, a unified foundation model for generalized multimodal learning. By constructing large-scale synthetic multimodal datasets with diverse correlation patterns, our model encodes transferable multimodal correlations during training and activates appropriate associations through in-context examples during inference. Extensive experiments on $18$ real-world datasets spanning $12$ modalities and $11$ prediction tasks demonstrate that our model achieves competitive performance with specialized models without task-specific adaptation.
