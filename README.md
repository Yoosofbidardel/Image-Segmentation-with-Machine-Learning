# Image-Segmentation-with-Machine-Learning
![image](https://user-images.githubusercontent.com/70627266/134719396-8132d8da-c08a-4ce6-9923-1c35cbf87a2d.png)


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
![image](https://user-images.githubusercontent.com/70627266/134719941-cb98e6dd-ec51-4685-8861-4b1d49a4479c.png)
RoI pool led to misalignments in getting the Region of Interest due to quantization of RoI coordinates. Since pixel-level segmentation required specificity hence authors of the Faster R-CNN cleverly solved it by implementing the RoI Align.

# Clone Mask R-CNN Github Repository
Now, primarily we download the architecture of the model which we are going to implement. Use the following command:

git clone: https://github.com/matterport/Mask_RCNN.git

# Pre Trained Weights
Since training a model takes hours and sometimes a day or more, hence it may not be feasible to train a model right now. Hence, we will utilize the pre-trained model to generate predictions on our input image.

Download Pretrained model from github: https://github.com/matterport/Mask_RCNN/releases


