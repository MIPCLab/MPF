# Weakly supervised semantic segmentation of SAR images via multi-level pseudo-label fusion

***Abstract:*** Pixel-level classification of synthetic aperture radar (SAR) images typically demands a substantial amount of labeled data for supervised learning. However, annotation of SAR image is quite difficult in practical scenes. Weakly supervised learning provides a novel training mode to significantly reduce pixel-level annotations by utilizing image-level labels. Despite this advantage, the generation of pseudo-labels in weakly supervised learning remains a challenge due to the complex scattering characteristics and spatial distributions of SAR data. To address this issue, a multi-level pseudo-label fusion approach is proposed for weakly supervised semantic segmentation of SAR images, which combines pixel gradient activation and inter-pixel relationship contrastive decision-making. Specifically, we design a pixel gradient activation module that uses multi-layer pixel gradients to highlight high-response regions, aiming to reduce the semantic confusion caused by complex backgrounds and enhance the confidence of pseudo-labels. Moreover, we introduce an inter-pixel relationship contrastive decision module to refine the integration of primary pseudo-labels across layers. This module improves the learning of pixel relationships within SAR images, addressing ambiguities caused by complex backgrounds and generating more accurate pseudo-labels. Experiments on two SAR datasets show that our approach significantly outperforms baseline methods and achieves new state-of-the-art (SoTA) performance in weakly supervised segmentation on the AIR-PolSAR-Seg benchmark with 37.46% mIoU, surpassing the leading weakly supervised method, ReCAM. It also yields superior results to other SOTA consistency-based methods on the EErDS-SAR dataset. The source code is available at https://github.com/MIPCLab/MPF.



## Citation

If you find this work valuable or use our code in your own research, please consider citing us: 

> Jie Geng, Yinju Nie, Ru Wang, Wen Jiang,
Weakly supervised semantic segmentation of SAR images via multi-level pseudo-label fusion,
ISPRS Journal of Photogrammetry and Remote Sensing,
Volume 231,
2026,
Pages 704-718,
ISSN 0924-2716,
https://doi.org/10.1016/j.isprsjprs.2025.11.020.
(https://www.sciencedirect.com/science/article/pii/S0924271625004599)


Bibtex format :

> @article{GENG2026704,
title = {Weakly supervised semantic segmentation of SAR images via multi-level pseudo-label fusion}, 
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {231},
pages = {704-718},
year = {2026},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2025.11.020},
url = {https://www.sciencedirect.com/science/article/pii/S0924271625004599},
author = {Jie Geng and Yinju Nie and Ru Wang and Wen Jiang},
keywords = {Weakly supervised learning, SAR image, Pixel gradient activation, Contrastive learning, Feature fusion}
}



## Dependencies

> torch=1.13.1 
> torchvision 
scipy 
numpy 
matplotlib 

If you encounter any problems, please feel free to open an issue.

