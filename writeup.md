# Self Driving Car Nanodegree: Poject 5 - Vehicle Detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Car_hog_color.png
[image2]: ./output_images/Noncar_hog_color.png
[image3]: ./output_images/TestImage_hog_color.png
[image4]: ./output_images/sliding_window_scale10.png
[image5]: ./output_images/sliding_window_scale15.png
[image6]: ./output_images/sliding_window_scale20.png
[image7]: ./output_images/sliding_window_scale25.png
[image8]: ./output_images/detections_10.png
[image9]: ./output_images/detections_15.png
[image10]: ./output_images/detections_20.png
[image11]: ./output_images/detections_25.png
[image12]: ./output_images/detections_10_2.png
[image13]: ./output_images/detections_15_2.png
[image14]: ./output_images/detections_20_2.png
[image15]: ./output_images/detections_25_2.png

[video1]: ./project_video_out.avi

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

Writeup outlines the methods I used for vehicle detection project. The implenented codes can be found in the Jupyter notebook file 'ProjectCode.ipynb'. For performance reasons I separated the video processing into a bare python file named 'ProcessVideo.py'.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used HOG features to obtain shape information. HOG extracts orientation information for orientations of gradients from the image, which is useful for classifying images. In addition to HOG features, I employed color features to help separate vehicles from non vehicles. Rather than using the color values directly I used a histogram to digitize the color information.

I used the pre-implemented code for HOG impelentation from _skimage_ library. This implementation allows us to choose the number of orientations, number of pixells per cell and number of cells per block. Similarly, I used histogram feature from _numpy_ library to extract color features. 

The first heading in the Jupyter notebook 'ProjectCode.ipynb' includes the code for HOG and color feature extraction. I experimented with test images and training data. My first intiution was that I should use the grayscale image for HOG feature extraction, so I used V value from HSV colorspace for HOG feature extraction, however in the final implementation I would change that (see next section for HOG parameters). For color features I extracted histograms from each channel and I concatenated them.

The below image is an example of features of a **car image**.
![alt text][image1]
The below image is an example of features of a **non-car image**
 ![alt text][image2]
 
 And below is an exaple of features extracted from a full frame.
![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I used a quantitative approach to selecting HOG parameters. I trained an SVM classifier and chose the parameters that maximized the the test accuracy of the classifier (see the following section for the choice and the training of the classifier).

My initial intiution was that for gradient information, I should use a grayscale image and the luminance/chrominance etc. should not include too much shape information. However I observed that using all channels instead of grayscale increase the accuracy about 2%. In addition, in my experiments using 'HSV' colorspace provided the best accuracy. In addition, using 12 orientations yielded the maximum performance. I observed 6 by 6 cells and 3 cels per block returned the best performance.

Here's the configuration that has provided the best accuracy for me.

    colorspace = 'HSV'
    orient = 12
    pix_per_cell = 6
    cell_per_block = 3
    hog_channel = "ALL"
    
    # parameters for Color features
    nbins=16
    bins_range=(0, 256)
    #Test accuracy 0.986204954955
    
The same configuration resulted in an accuracy from 0.986204954955, to 0.991554054054 so we have ±5%  precision when measuring the error rate.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

To train the classifier I used the training dataset that is provided in the course material. We have a large number of samples to train our classifier. For this reason, my classifier of choice is SVM because SVM classifiers yield satisfactory results with large numbers of data.

I used the implementation of SVM from _sklearn_ library. I experimented and found that linear SVM classifier yielded satisfactory results. The 'Part 2' heading of the 'ProjectCode.ipynb' includes the instantiation and training of the classifier.

I extracted the HOG and color features from the training dataset. Then I used a scaler to scale the features so that the feature, that is strongest in amplitude would not dominate the training. After that, I split the samples where I used 80% of the samples for training the classifier and the remaining 20% of the samples for testing.

I trained my classifier several times varying the parameters for HOG and color feature extraction. At the end of training I choose the best parameters that maximized the testing accuracy.

At the end of the training I saved the trained classifier and scaler in order not to train the classifier each time I run the software.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search with overlapping windows, so that I could classify each window. The 'slide_windows()' function of 'Part 3' heading of the 'ProjectCode.ipynb' demonstrates a sliding window extraction procedure.

I choose four scales to implement the sliding windows. And instead of searching whole image that includes places logically cars cannot exist, I limited each scale to where cars of these sizes are likely to exist. The following four pictures demostrate the scales and search regions.

    scale = 1.0
    y_start_stop = (380, 500)
![alt text][image4]

    scale = 1.5
    y_start_stop = (390, 550)
![alt text][image5]

    scale = 2.0
    y_start_stop = (400, 580)
![alt text][image6]


    scale = 2.5
    y_start_stop = (500, 680)
![alt text][image7]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using HOG features extracted from all three channels from image, that is converted to the HSV colorspace and color histograms of each channel. Instead of calculating HOG features everytime, I run HOG extraction once, and use its sub-windows for classification for each scale. 

The 'Classification' part of 'Part 3' heading of the 'ProjectCode.ipynb' shows the code I used for sub-sampling HOG and running classification on sliding windows.


 Here are some example images for scales 1.0, 1.5, 2.0, and 2.5:

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.avi)
![alt text][video1]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

--Aspect ratio of the images, and search window shape

--Thresholding of heatmaps is sub-optimal / morphologica erosion/dilation

--Rotation invariance
