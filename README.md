# ESarDet
## Paper

papar is available at: https://www.mdpi.com/2072-4292/15/12/3018

## Overall structure

![Image text](https://github.com/ZYMCCX/ESarDet/blob/main/structure.png)

## Abstract 

Ship detection using synthetic aperture radar (SAR) has been extensively utilized in both the military and civilian fields. On account of complex backgrounds, large scale variations, small scale targets, and other challenges, it is difficult for current SAR ship detection methods to strike a balance between detection accuracy and computation efficiency. To overcome aforementioned challenges, ESarDet, an efficient SAR ship detection method based on context information and large effective receptive field (ERF), is proposed. We introduce the anchor-free object detection method YOLOX-tiny as baseline model and make several improvements on it. First, a lightweight context information extraction module based on large kernel convolution and attention mechanism, context attention auxiliary network (CAA-Net), is constructed to expand ERF and enhance the merger of context and semantic information. Second, to expand ERF and prevent the loss of semantic information, atrous attentive spatial pyramid pooling fast (A2SPPF) is proposed to substitute the spatial pyramid pooling (SPP) in YOLOX-tiny. Finally, a novel convolution block, atrous attentive cross stage partial layer (A2CSPlayer) is designed to enhance the fusion of feature maps from various scales. Extensive experiments are carried out on three public SAR ship datasets, DSSDD, SSDD, and HRSID, to verify the effectiveness of the proposed ESarDet. On the one hand, we conduct ablation experiments to verify the validity of each proposed module. On the other hand, we perform comparison experiments in which we compare the proposed ESarDet to other state-of-the-art (SOTA) detectors. Our proposed ESarDet achieves 97.93%, 97.96% and 93.22% in the average precision (AP) metrics for DSSDD, SSDD and HRSID datasets, respectively, while the number of parameters of the model is only 6.2 M, outperforming other SOTA detectors. The experimental results indicate that the proposed ESarDet can detect ships in SAR images with high efficiency and precision.
