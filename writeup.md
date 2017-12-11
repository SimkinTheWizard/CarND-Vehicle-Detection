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
[image16]: ./output_images/heat_map.png
[image17]: ./output_images/thresholded_heat_map.png
[image18]: ./output_images/bounding_boxes_on_image.png
[image19]: ./output_images/small_artifacts.png
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

For performance reasons, I separated video processing to a separate file and run it on bare python instead of Jupyter notebook. Please see 'ProcessVideo.py' for video implementation code.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The operations return multiple detections on the detected images and some false positives, where there are no cars. To prevent/reduce this effect I used heat maps. Heat maps work with the assumption that true positives will be seen in multiple overlapping cells, and the cells where there are no detections in the neighboring cells may be false positives. 

For each detection on image, I added the detection to the heat map. After all detections are added I used a threshold on the heat map. This thresholding eliminated parts of the image where only one detection is present. After this stepp I separated the detections with connected components labelling and received their bounding boxes. Finally, I have drawn the bounding boxes as the final detections.

The 'Heat Map' part of 'Part 3' heading of the 'ProjectCode.ipynb' shows the code I used heat maps and labeling.



### Here are detections:
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

### The heat maps of these detections are dislplayed in the following heat map:
![alt text][image16]

### The thresholding eliminated the false positive:
![alt text][image17]

### Here the resulting bounding boxes are drawn onto the frame:
![alt text][image18]

**Update:** After these operations my code was still having many false positives, to filter out these false positives I increase the search span of all of the scales in multi-scale search windows. This way true positives would have more consensus amongs many windows, and I would be able to use a greater threshold. I also observed that the false positives were smaller and closer to the bottom. This small objects could not be at that closer to the bottom so I limited the search space of the smaller windows to the upper sides of the screen. 

However increasing the threshold of heatmaps has a result of increasing the small artifacts remaining after thresholding the heatmap.
![alt text][image15]

 To get around this artifacts I added a morphological closing operation to thresholding function. So the final function has become the following:
 
     def apply_threshold(heatmap, threshold):
        # Use morphological filter filter discontinuities
        kernel = np.ones((8, 8), np.uint8)
        heatmap = cv2.morphologyEx(heatmap, cv2.MORPH_CLOSE, kernel)
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

This operation reduced the unwanted artifacts significantly.
 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are some points I've observed that degrades the performance of the system, and I share my opinions on how to improve them.

The first observation is that our classification consisted of square shaped images whereas more often than not, test samples are not squares but rectangles. This effect of aspect ratio is decreasing our classification performance. Althoug it is not practical train classifiers and make inferences with different shapes, we can use asymmetrical scaling when doing sliding video search. In other words we can scale in x direction with a different ratio than y direction. We are already doing multiple scaling operations, so this would not require too much effort, but it may increase the performance.

Another point I've observed is that thresholding heatmaps is not optimal. Sometimes thresholding the heatmap produces little island shapes where detections overlap but not too much. A better way to handle heatmaps may be to use grayscale morphological operations. A series of grayscale erosion and dilation operations used standalone or together with thresholding would eliminate these island shapes, and provide better shapes, better resembling the detected vehicles. 

