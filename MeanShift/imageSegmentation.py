from scipy.spatial.distance import cdist
import scipy.io
import tqdm
import sys
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import lab2rgb, rgb2lab

# from MeanShift.plotclusters3D import *
# from MeanShift.histogram import *

from time import time

import cv2


def histogram(image):
    """
    Does histogram equalization by changing to color scheme to BGR, then to YUV.
    After doing the histogram equalization in the YUV color scheme, transform back to
    BGR and then RGB. Output this image.
    Hints have been used by looking at this link:
    https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image

    Args:
        image: image in RGB in the data folder.

    Output:
        Equalized image in RGB
    """
    img = cv2.imread(f"./Data/{image}", cv2.IMREAD_COLOR)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # transform yuv to bgr
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # transform bgr to rgb
    img_output_rgb = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)

    return img_output_rgb



def plotclusters3D(data, labels, peaks):
    """
    Plots the modes of the given image data in 3D by coloring each pixel
    according to its corresponding peak.

    Args:
        data: image data in the format [number of pixels]x[feature vector].
        labels: a list of labels, one for each pixel.
        peaks: a list of vectors, whose first three components can
        be interpreted as BGR values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    bgr_peaks = np.array(peaks[:, 0:3], dtype=float)
    rgb_peaks = bgr_peaks[...,::-1]
    rgb_peaks /= 255.0
    for idx, peak in enumerate(rgb_peaks):
        color = np.random.uniform(0, 1, 3)
        #TODO: instead of random color, you can use peaks when you work on actual images
        # color = peak
        points = np.where(labels == idx)[:]
        cluster = data[points].T
        ax.scatter(cluster[0], cluster[1], cluster[2], c=[color], s=.5)
    # plt.savefig(filename)
    plt.show()

def find_peak(data, idx, r, t=0.01):
    """
    Find the peak of a datapoint according to the Mean Shift Algorithm
    First calculate mean of datapoint, according to circle with radius r.
    Then move the middle point of the circle to the calculated mean,
    and calculate the new mean. Keep doing this until the distance between two means
    is less than threshold t.

    Args:
        data: the entire data set
        idx: the column index for which you want to calculate the mean
        r: the radius of the circle to use to calculate the mean
        t: the cut off point threshold for the distance between two means

    Output:
        calculated peak
    """
    # Get the distances for all other data points using Euclidean metric
    distances = cdist(data[idx].reshape(1, -1), data, metric='euclidean')
    # Take only the points that are in the circle with radius r
    points = np.argwhere(distances <= r)[:, 1]
    # calculate the mean of these points
    peak = np.mean(data[points[:]], axis=0)
    distance = 1000
    # Continue doing the steps above until the distance between two means is
    # less than t
    while distance > t:
        old_peak = peak
        # Calculate distance between previous calculated peak and all datapoints
        distances = cdist(old_peak.reshape(1, -1), data, metric='euclidean')
        # Get all points that lie in the circle with radius r
        points = np.argwhere(distances <= r)[:, 1]
        # Calculate the new peak
        peak = np.mean(data[points[:]], axis=0)
        # Calculate the distance between the previous peak and the new peak
        distance = np.linalg.norm(old_peak - peak)

    return peak


def mean_shift(data, r):
    """
    Use the findpeak function to calculate the peaks of all data points.
    If the calculated peak is within a certain range r/2 of an existing peak,
    merge the two peaks together.

    Args:
        data: data you want to do mean shift algorithm on
        r: radius of the circle to consider all the data points in to calculate peak

    Output:
        labels: For every data point a label associated with a certain peak
        peaks: All calculated peaks
    """
    # initialize the labels for the data
    labels = np.zeros(len(data))
    peaks = []
    number_of_labels = 0
    label_peak = {}
    # Calculate the peak for every data point in data
    for i, point in enumerate(tqdm.tqdm(data)):
        peak = find_peak(data, i, r)
        # If a peak exists, check if there is an existing peak within
        # a certain distance, if so merge them together and give the current datapoint
        # the same label, otherwise, add the peak to the list of peaks and give a
        # a label
        if len(peaks) != 0:
            peakDistance = cdist(peak.reshape(1, -1), np.array(peaks))
            points = np.argwhere(peakDistance < r / 2)[:, 1]
            if len(points) != 0:
                labels[i] = points[0]
            else:
                peaks.append(peak)
                labels[i] = number_of_labels
                label_peak[number_of_labels] = peak
                number_of_labels += 1
        else:
            peaks.append(peak)
            labels[i] = number_of_labels
            label_peak[number_of_labels] = peak
            number_of_labels += 1

    return labels, np.array(peaks)


def find_peak_opt(data, idx, r, t=0.01, c=4):
    """
       Find the peak of a datapoint according to the Mean Shift Algorithm
       First calculate mean of datapoint, according to circle with radius r.
       Then move the middle point of the circle to the calculated mean,
       and calculate the new mean. Keep doing this until the distance between two means
       is less than threshold t.
       If a data point lies within r/c after having calculated a peak, it will
       be labelled that the data point lies within the search path of a data point.
       Args:
           data: the entire data set
           idx: the column index for which you want to calculate the mean
           r: the radius of the circle to use to calculate the mean
           t: the cut off point threshold for the distance between two means
           c: the parameter that determines how big the search path is.
       Output:
           peak: calculated peak
           cpts: a list whether a data point lies in the search path or not
       """
    # initialize the list that tells us if a data point lies in the search path
    cpts = np.zeros(len(data))
    # Get the distances for all other data points using Euclidean metric
    distances = cdist(data[idx].reshape(1, -1), data, metric='euclidean')
    # Take only the points that are in the circle with radius r
    points = np.argwhere(distances <= r)[:, 1]
    # Calculate the mean of these points
    peak = np.mean(data[points[:]], axis=0)
    # Calculate the distance between all points and the peak
    search_path_distance = cdist(peak.reshape(1, -1), data, metric='euclidean')
    # Get all the points that lie within r/c
    search_path_points = np.argwhere(search_path_distance <= r / c)[:, 1]
    # Mark the above points as being in the search path
    cpts[search_path_points[:]] = 1
    distance = 1000
    # Repeat the steps above until the difference between two means is less
    # than threshold t
    while distance > t:
        old_peak = peak
        distances = cdist(old_peak.reshape(1, -1), data, metric='euclidean')
        points = np.argwhere(distances <= r)[:, 1]
        peak = np.mean(data[points[:]], axis=0)
        search_path_distance = cdist(peak.reshape(1, -1), data, metric='euclidean')
        search_path_points = np.argwhere(search_path_distance <= r / c)[:, 1]
        cpts[search_path_points[:]] = 1
        distance = np.linalg.norm(old_peak - peak)
    return peak, cpts


def mean_shift_opt(data, r, threshold, c, basin=False, path=False):
    """
    Use the findpeak function to calculate the peaks of all data points.
    If the calculated peak is within a certain range r/2 of an existing peak,
    merge the two peaks together. There are two options for speeding the algorithm up:
    the basin of attraction speedup and the search path speedup.
    Args:
        data: data you want to do mean shift algorithm on
        r: radius of the circle to consider all the data points in to calculate peak
        threshold: the threshold between moving peaks
        c: the search path parameter for the second speed up
        basin: Boolean - Whether you want to use the basin of attraction speedup
        path: Boolean - Whether you want to use the search path speedup
    Output:
        labels: For every data point a label associated with a certain peak
        peaks: All calculated peaks
    """
    # Initialize the labels list for all parameters
    labels = np.full(len(data), -1)
    peaks = []
    number_of_labels = 0
    label_peak = {}
    # Iterative over all data points to calculate the peak
    for i, point in enumerate(tqdm.tqdm(data)):
        # Due to the speed ups, it is possible that a data point already
        # has a label, so we can skip this data point to calculate its peak
        if labels[i] == -1:
            # if you want to use the search path speed up
            if path:
                # Calculate the peak
                peak, cpts = find_peak_opt(data, i, r, threshold, c)
                # Find all points that lie on the search path of the previous
                # calculated mean
                search_path = np.argwhere(cpts == 1)
                # If a peak exists, check if there is an existing peak within
                # a certain distance r/2, if so merge them together and give the current datapoint
                # the same label, otherwise, add the peak to the list of peaks and give a
                # a label
                if len(peaks) != 0:
                    peakDistance = cdist(peak.reshape(1, -1), np.array(peaks))
                    points = np.argwhere(peakDistance < r / 2)[:, 1]
                    if len(points) != 0:
                        labels[i] = points[0]
                        # Label all the points that lie in the search path
                        labels[search_path[:]] = points[0]
                    else:
                        peaks.append(peak)
                        # Checking if you want to use the basin of attraction speedup
                        if basin:
                            # Calculate the distance to all data points with the peak in the as the
                            # point to calculate from
                            distances = cdist(peak.reshape(1, -1), data, metric='euclidean')
                            # Get all points that lie in the circle with radius r
                            points = np.argwhere(distances <= r)[:, 1]
                            # Label all the points that lie in the circle associated with this peak
                            labels[points[:]] = number_of_labels
                        # Label all the points that lie in the search path
                        labels[search_path[:]] = number_of_labels
                        labels[i] = number_of_labels
                        label_peak[number_of_labels] = peak
                        number_of_labels += 1
                else:
                    peaks.append(peak)
                    # Checking if you want to use the basin of attraction speedup
                    if basin:
                        # Calculate the distance to all data points with the peak in the as the
                        # point to calculate from
                        distances = cdist(peak.reshape(1, -1), data, metric='euclidean')
                        # Get all points that lie in the circle with radius r
                        points = np.argwhere(distances <= r)[:, 1]
                        # Label all the points that lie in the circle associated with this peak
                        labels[points[:]] = number_of_labels
                    # Label all the points that lie in the search path
                    labels[search_path[:]] = number_of_labels
                    labels[i] = number_of_labels
                    label_peak[number_of_labels] = peak
                    number_of_labels += 1
            else:
                # You do not want to use the search path speedup
                peak = find_peak(data, i, r)
                # If a peak exists, check if there is an existing peak within
                # a certain distance r/2, if so merge them together and give the current datapoint
                # the same label, otherwise, add the peak to the list of peaks and give a
                # a label
                if len(peaks) != 0:
                    peakDistance = cdist(peak.reshape(1, -1), np.array(peaks))
                    points = np.argwhere(peakDistance < r / 2)[:, 1]
                    if len(points) != 0:
                        labels[i] = points[0]
                    else:
                        peaks.append(peak)
                        # Checking if you want to use the basin of attraction speedup
                        if basin:
                            # Calculate the distance to all data points with the peak in the as the
                            # point to calculate from
                            distances = cdist(peak.reshape(1, -1), data, metric='euclidean')
                            # Get all points that lie in the circle with radius r
                            points = np.argwhere(distances <= r)[:, 1]
                            # Label all the points that lie in the circle associated with this peak
                            labels[points[:]] = number_of_labels
                        labels[i] = number_of_labels
                        label_peak[number_of_labels] = peak
                        number_of_labels += 1
                else:
                    peaks.append(peak)
                    # Checking if you want to use the basin of attraction speedup
                    if basin:
                        # Calculate the distance to all data points with the peak in the as the
                        # point to calculate from
                        distances = cdist(peak.reshape(1, -1), data, metric='euclidean')
                        # Get all points that lie in the circle with radius r
                        points = np.argwhere(distances <= r)[:, 1]
                        # Label all the points that lie in the circle associated with this peak
                        labels[points[:]] = number_of_labels
                    labels[i] = number_of_labels
                    label_peak[number_of_labels] = peak
                    number_of_labels += 1
    return labels, np.array(peaks)


def imSegment(im, r, threshold=0.01, c=4, basin_of_attraction=True, path_speedup=True, feature_type='3D'):
    """
    The function to be called to get the segmented image

    Args:
        im: The image you want to segment
        r: the radius of the circle for calculating peaks
        threshold: the threshold of distance between two moving peaks
        c: the parameter that determines the distance for a point to lie in the searc path
        basing_of_attraction: Boolean - whether you want to use this speedup
        path_speedup: Boolean - whether you want to use the search path speedup
        feature_type: If you want to use 3D or 5D image to segment on

    Output:
        segmented_image_rgb: The segmented image in rgb
        labels: the labels for all the data points
    """
    # Transform color space to LAB
    imlab = rgb2lab(im)
    if feature_type == '5D':
        # if featureType is 5D --> add coordinates
        xCoords = np.expand_dims(np.tile(np.array(range(im.shape[0])).reshape(-1, 1), reps=im.shape[1]),
                                 axis=-1)
        yCoords = np.expand_dims(np.tile(np.array(range(im.shape[1])).reshape(-1, 1), reps=im.shape[0]).T,
                                 axis=-1)
        imlab = np.append(imlab, xCoords, axis=-1)
        imlab = np.append(imlab, yCoords, axis=-1)
        flattened_image = imlab.reshape(-1, 5)
    else:
        flattened_image = imlab.reshape(-1, 3)
    # If you want to use a speed up, use the correct mean shift function
    if basin_of_attraction or path_speedup:
        labels, peaks = mean_shift_opt(flattened_image, r, threshold, c, basin_of_attraction, path_speedup)
        # This can be run, but will still how random colors, I had some error and didn't have time to
        # fix it
        # plotclusters3D(im.reshape(-1,3), labels, peaks[:, 0:3].T)
    else:
        labels, peaks = mean_shift(flattened_image, r)
        # This can be run, but will still how random colors, I had some error and didn't have time to
        # fix it
        # plotclusters3D(im.reshape(-1,3), labels, peaks[:, 0:3].T)

    # Get the segmented values based on the labels and peaks
    segemented_values = np.zeros((len(labels), 3))
    # If you have a 5D image, go back to 3D rgb
    peaks = peaks[:, 0:3]
    for idx, x in enumerate(peaks):
        points = np.argwhere(labels == idx)[:, 0]
        segemented_values[points[:]] = x
    print(f'number of unique labels {np.unique(labels)}')
    print(f'number of peaks {len(peaks)}')
    # Get the same shape as the input shape
    segmented_image = segemented_values.reshape(im.shape)
    # Transform back to rgb color scheme
    segmented_image_rgb = lab2rgb(segmented_image)
    return segmented_image_rgb, labels


def segmentation(argv):
    print(argv)
    # Image you want to segment, the image is assumed to be in the Data folder
    picture = argv[0]
    # The radius you want to use
    radius = float(argv[1])
    # The threshold you want to use
    threshold = float(argv[2])
    # The c parameter you want to use
    c = float(argv[3])
    # If you want to use the basin of attraction or search path speedup
    if argv[4] == 'True':
        basin = True
    else:
        basin = False
    if argv[5] == 'True':
        search_path = True
    else:
        search_path = False
    # Which feature type you want to use
    feature_type = (argv[6])
    # If you want to use histogram equalization
    if argv[7] == 'True':
        preprocessing = True
    else:
        preprocessing = False
    if preprocessing:
        image = plt.imread(f"./Data/{picture}")
        plt.subplot(141)
        plt.title("Original image")
        plt.imshow(image)
        plt.axis('off')

        start_time = time()
        segmented_image_3d, labels_3d = imSegment(image, radius, threshold=threshold, c=c, basin_of_attraction=basin,
                                                  path_speedup=search_path, feature_type=feature_type)
        print(f"Took {time() - start_time} seconds")
        plt.subplot(142)
        plt.title(f"Segmented")
        plt.imshow(segmented_image_3d)
        plt.axis('off')

        equalized_image = histogram(picture)
        plt.subplot(143)
        plt.title("Histogram")
        plt.axis('off')
        plt.imshow(equalized_image)

        start_time = time()
        segmented_image_3d_eq, labels_3d_eq = imSegment(equalized_image, radius, threshold=threshold, c=c,
                                                        basin_of_attraction=basin,
                                                        path_speedup=search_path, feature_type=feature_type)
        print(f"Took {time() - start_time} seconds")
        plt.subplot(144)
        plt.axis('off')
        plt.title("Histo segmented")
        plt.imshow(segmented_image_3d_eq)

        plt.show()
    else:
        image = plt.imread(f"./Data/{picture}")
        plt.subplot(121)
        plt.title("Original image")
        plt.imshow(image)
        plt.axis('off')

        start_time = time()
        segmented_image_3d, labels_3d = imSegment(image, radius, threshold=threshold, c=c, basin_of_attraction=basin,
                                                  path_speedup=search_path, feature_type=feature_type)
        print(f"Took {time() - start_time} seconds for 3D")
        plt.subplot(122)
        plt.title(f"Segmented image")
        plt.imshow(segmented_image_3d)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 1:
        segmentation(sys.argv[1:])
    else:
        parameters = ['bulbasaur.jpg', 20, 0.01, 4, "True", "True", '3D', "True"]
        segmentation(parameters)
