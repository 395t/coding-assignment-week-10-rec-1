This is the coding assignment for Mask R-CNN, which was not successfully done. I used an open source code in the following link, I modified the code a little bit, and trained the Mask R-CNN model on VOC_2012, VOC_2007 datasets. <br/>https://bjornkhansen95.medium.com/mask-r-cnn-for-segmentation-using-pytorch-8bbfa8511883 <br/>

However, when the code was running on Colab under GPU/High RAM settings, there was always one error saying that an illegal cuda memory was encountered. If I change the setting to CPU, the program runs smoothly but extremely slow. Therefore, I only trained my Mask R-CNN model on a small portion of VOC_2012 and VOC_2007 datasets for two epoches respectively. The following pictures are my results: (the first two are training loss on VOC_2007, and the last two are training loss on VOC_2012)<br/>

<p align="center">
  <img src="mask_r_cnn_voc2007.png" width="350" title="VOC2007">
  <img src="mask_r_cnn_voc2007_2.png" width="350">
</p>

<p align="center">
  <img src="mask_r_cnn_voc2012.png" width="350" title="VOC2007_1">
  <img src="mask_r_cnn_voc2012_1.png" width="350">
</p>

I also tested my own picture on pretrained Mask R-CNN model provided by pytorch, here are the results <br/>

<p align="center">
  <img src="WIN_20211023_23_02_32_Pro.jpg" width="350" title="VOC2007_1">
  <img src="mask_r_cnn_coco.png" width="350">
</p>
