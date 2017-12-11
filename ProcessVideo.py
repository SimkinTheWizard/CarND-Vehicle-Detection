import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from skimage.feature import hog
from scipy.ndimage.measurements import label

from sklearn.externals import joblib
from joblib import Parallel, delayed
import multiprocessing
import gc

gc.collect()
# we will load the model we trained before
clf = joblib.load('classifier.pkl')
X_scaler = joblib.load('scale.pkl')

video_file="project_video.mp4"
output_file="project_video_out6.avi"

verbose=False
multi_core=True
CPU_BATCH_SIZE=128
heat_threshold = 5

def cvt_colorspace(image, cspace='RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)
    return feature_image


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     feature_vec=True):
    vis = False
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                   visualise=vis, feature_vector=feature_vec)
    return features


def find_cars(img, y_start_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              window, colorspace, draw=False):
    found_windows = []
    if draw:
        draw_img = np.copy(img)
    img2 = cvt_colorspace(img, colorspace)

    img_cropped = img2[y_start_stop[0]:y_start_stop[1], :, :]

    if scale is not 1:
        imshape = img_cropped.shape
        img_cropped = cv2.resize(img_cropped, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = img_cropped[:, :, 0]
    ch2 = img_cropped[:, :, 1]
    ch3 = img_cropped[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # print (nxsteps)
    # print (nysteps)
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(img_cropped[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            # spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((hog_features, hist_features,)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                found_windows.append(((xbox_left, ytop_draw + y_start_stop[0]),
                                      (xbox_left + win_draw, ytop_draw + win_draw + y_start_stop[0])))
                if (draw):
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + y_start_stop[0]),
                                  (xbox_left + win_draw, ytop_draw + win_draw + y_start_stop[0]), (0, 0, 255), 6)
    if not draw:
        return found_windows
    return found_windows, draw_img




def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Use morphological filter filter discontinuities
    kernel = np.ones((8, 8), np.uint8)
    heatmap = cv2.morphologyEx(heatmap, cv2.MORPH_CLOSE, kernel)
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def connetcted_components_labeling(heatmap):
    labels = label(heatmap)
    rectangles = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        rectangles.append(bbox)
    return labels, rectangles


def draw_rectangles_on_image(img, rectangles):
    img2 = img.copy()
    #print (rectangles)
    for rectangle in rectangles:
        cv2.rectangle(img2, rectangle[0], rectangle[1], (0, 0, 255), 2)
    return img2


def process_frame(img):
    heat_map = np.zeros_like(img[:, :, 0])

    scale = 1
    w_size = 64
    y_start_stop = (380, 450)
    debug_rectangles=[]
    found_windows = find_cars(img, y_start_stop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block,
                              0, nbins, w_size, colorspace, False)
    add_heat(heat_map, found_windows)
    for window in found_windows :
        debug_rectangles.append(window)

    scale = 1.5
    y_start_stop = (380, 550)
    found_windows = find_cars(img, y_start_stop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block,
                              0, nbins, w_size, colorspace, False)
    add_heat(heat_map, found_windows)
    for window in found_windows :
        debug_rectangles.append(window)

    scale = 2.0
    y_start_stop = (380, 580)
    found_windows = find_cars(img, y_start_stop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block,
                              0, nbins, w_size, colorspace, False)
    add_heat(heat_map, found_windows)
    for window in found_windows :
        debug_rectangles.append(window)

    scale = 2.5
    y_start_stop = (380, 680)
    found_windows = find_cars(img, y_start_stop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block,
                              0, nbins, w_size, colorspace, False)
    add_heat(heat_map, found_windows)
    for window in found_windows :
        debug_rectangles.append(window)
    final_heat = apply_threshold(heat_map, heat_threshold)
    labels, rectangles = connetcted_components_labeling(final_heat)
    rectangle_image = draw_rectangles_on_image(img, rectangles)
    #rectangle_image = draw_rectangles_on_image(img,debug_rectangles)
    return rectangle_image





colorspace = 'HSV'
orient = 12
pix_per_cell = 6
cell_per_block = 3
# we will use all of the channels for HOG features
# hog_channl = "ALL"

# parameters for Color features
nbins = 16
bins_range = (0, 256)





print("Opening video file for processing")
capture = cv2.VideoCapture(video_file)
length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print( "The video has " + str(length) + " frames" )



# Define the codec and create VideoWriter object

fps = 30

if verbose:
    capSize = (1280, 900)
else:
    capSize = (1280, 720)
# For ubuntu
#fourcc = cv2.VideoWriter_fourcc('x', '2', '6', '4')
# For mac
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter()
success = writer.open(output_file,fourcc,fps,capSize,True)
if not success:
    print("Could not open output video file for writing")
    exit(-1)




frame_count = 0
num_cores = multiprocessing.cpu_count()
print ("This cpu has "+str(num_cores)+" cores")


batch_frames = []
while(capture.isOpened()):
    ret, img = capture.read()
    if ret==True:
        frame_count = frame_count + 1
        print("Frame " + str(frame_count) + " of " + str(length) + "\r")

        #if frame_count < 1033:
        #    continue
        if (multi_core is not True):
            output = process_frame(img)
            writer.write(output)
            cv2.imshow('frame', output)
        else:
            batch_frames.append(img)
            if (len(batch_frames) >= CPU_BATCH_SIZE):
                results = Parallel(n_jobs=num_cores)(delayed(process_frame)(batch_frames[i]) for i in range(len(batch_frames)))
            #    results = Parallel(                delayed(process_frame)(batch_frames[i]) for i in range(len(batch_frames)))
                print ("Processed " + str(len(results)) + "frames")
                for i in range( len(batch_frames)):
                    output = np.asarray(results[i])
                    writer.write(output)
                    cv2.imshow('frame',results[i])
                    cv2.waitKey(1)
                batch_frames = []
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

print()
print("Finish")

# Release everything if job is finished
capture.release()
writer.release()