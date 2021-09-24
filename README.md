# Image-Segmentation-with-Machine-Learning

# Mask R-CNN
We are going to perform image segmentation using the Mask R-CNN architecture. It is an extension of the Faster R-CNN Model which is preferred for object detection tasks.

The Mask R-CNN returns the binary object mask in addition to class label and object bounding box. Mask R-CNN is good at pixel level segmentation.


1. Feature Extraction
We utilize the ResNet 101 architecture to extract features from the input image. As a result, we get feature maps which are transmitted to Region Proposed Network

2. Region Proposed Network (RPN)
After obtaining the feature maps, bounding box candidates are determined and thus RPN extracts RoI (Region of Interest)

3. RoI Pool
Faster R-CNN uses an RoI Pool layer to compute features from the obtained proposals in order to infer the class of the object and bounding box coordinates.

4. RoI Align
