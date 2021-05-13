from scipy.spatial.distance import cdist
import scipy.io
import tqdm

from skimage.color import lab2rgb, rgb2lab

from MeanShift.plotclusters3D import *
from MeanShift.histogram import *

from time import time


def findpeak1D(data, idx, r):
    distance = cdist(data[idx].reshape(-1, 1), data, metric='euclidean')
    points = np.argwhere(distance <= r)[:, 1]
    peak = np.mean(data[points[:]], axis=0)
    return peak


def findpeak3D(data, idx, r):
    distance = cdist(data[idx].reshape(1, -1), data, metric='euclidean')
    points = np.argwhere(distance <= r)[:, 1]
    peak = np.mean(data[points[:]], axis=0)
    # peak = np.mean(data[(distance <= r).T.reshape(-1)], axis=0)
    return peak


def findpeak_new(data, idx, r):
    distance = cdist(idx.reshape(1, -1), data, metric='euclidean')
    points = np.argwhere(distance <= r)[:, 1]
    peak = np.mean(data[points[:]], axis=0)
    # peak = np.mean(data[(distance <= r).T.reshape(-1)], axis=0)
    return peak


def mean_shift1D(data, r):
    labels = np.arange(len(data))
    peaks = np.copy(data)
    t = 0.01

    optimal = False
    while not optimal:
        oldPeaks = np.copy(peaks)
        for i, point in enumerate(tqdm.tqdm(oldPeaks)):
            label = labels[i]
            peak = findpeak1D(oldPeaks, i, r)
            peakDistance = cdist(peak.reshape(1, 1), oldPeaks.reshape(-1, 1))
            points = np.argwhere(peakDistance <= r / 2)[:, 1]
            for x in points:
                if x != i:
                    label = labels[x]
                    peak = peaks[x]
                    break
            labels[i] = label
            peaks[i] = peak

            peakMovements = np.sum(np.abs(oldPeaks - peaks))
            if peakMovements < t:
                optimal = True
    return labels, peaks


def mean_shift3D(data, r, basin_of_attraction=True):
    labels = np.arange(len(data))
    peaks = np.copy(data)
    t = 0.01
    runs = 0
    optimal = False
    while not optimal:
        oldPeaks = np.copy(peaks)
        toLookAt = np.ones(len(data), dtype=bool)
        for i, point in enumerate(tqdm.tqdm(data)):
            if not toLookAt[i]:
                continue

            label = labels[i]
            peak = findpeak3D(data, i, r)
            peakDistance = cdist(peak.reshape(1, -1), peaks)
            points = np.argwhere(peakDistance < r / 2)[:, 1]
            # If there exist a point that is closer than r/2, merge the two peaks
            for x in points:
                if x != i:
                    label = labels[x]
                    peak = peaks[x]
                    break
            labels[i] = label
            peaks[i] = peak

            if basin_of_attraction:
                peakDistance = cdist(peak.reshape(1, -1), peaks)
                points = np.argwhere(peakDistance <= r)[:, 1]
                peaks[points[:]] = peak
                labels[points[:]] = label
                toLookAt[points[:]] = False
        runs += 1
        peakMovements = np.sum(np.abs(oldPeaks - peaks))
        print(f"Run {runs} with peakmovements: {peakMovements}")
        if peakMovements < t:
            optimal = True
    return labels, peaks


def new_meanshift(data, r):
    labels = np.arange(len(data))
    peak_values = np.copy(data)
    peaks = np.copy(data)
    t = 0.01
    runs = 0
    optimal = False
    while not optimal:
        oldPeaks = np.copy(peaks)
        old_peak_values = np.copy(peak_values)
        old_labels = labels
        peaks = []
        new_labels = []
        for i, point in enumerate(tqdm.tqdm(oldPeaks)):
            existing_peak = False
            old_label = old_labels[i]
            new_label = old_labels[i]
            new_peak = findpeak_new(data, point, r)
            if (len(peaks)) != 0:
                peakDistance = cdist(new_peak.reshape(1, -1), peaks)
                points = np.argwhere(peakDistance < r / 2)[:, 1]
                # If there exist a point that is closer than r/2, merge the two peaks
                for x in points:
                    if x != i:
                        new_label = new_labels[x]
                        new_peak = peaks[x]
                        existing_peak = True
                        break
            if not existing_peak:
                peaks.append(new_peak)
                new_labels.append(old_label)
            # if new_label != old_label:
            #     old_label = new_label
            points = np.argwhere(old_labels == old_label)
            labels[points[:]] = new_label
            peak_values[points[:]] = new_peak

        peakMovements = np.sum(np.abs(old_peak_values - peak_values))
        print(f"Run {runs} with peakmovements: {peakMovements}")
        if peakMovements < t:
            optimal = True
    return labels, peaks


def find_peak(data, idx, r, t=0.01):
    distances = cdist(data[idx].reshape(1, -1), data, metric='euclidean')
    points = np.argwhere(distances <= r)[:, 1]
    peak = np.mean(data[points[:]], axis=0)
    distance = 1000
    while distance > t:
        old_peak = peak
        distances = cdist(old_peak.reshape(1, -1), data, metric='euclidean')
        points = np.argwhere(distances <= r)[:, 1]
        peak = np.mean(data[points[:]], axis=0)
        distance = np.linalg.norm(old_peak - peak)

    return peak


def mean_shift(data, r):
    labels = np.zeros(len(data))
    peaks = []
    number_of_labels = 0
    label_peak = {}
    peak_values = np.copy(data)
    for i, point in enumerate(tqdm.tqdm(data)):
        peak = find_peak(data, i, r)
        if len(peaks) != 0:
            peakDistance = cdist(peak.reshape(1, -1), np.array(peaks))
            points = np.argwhere(peakDistance < r / 2)[:, 1]
            if len(points) != 0:
                labels[i] = points[0]
                peak_values[i] = label_peak[points[0]]
            else:
                peaks.append(peak)
                labels[i] = number_of_labels
                label_peak[number_of_labels] = peak
                peak_values[i] = peak
                number_of_labels += 1
        else:
            peaks.append(peak)
            labels[i] = number_of_labels
            label_peak[number_of_labels] = peak
            peak_values[i] = peak
            number_of_labels += 1
    return labels, np.array(peaks)


def find_peak_opt(data, idx, r, t=0.01, c=4):
    cpts = np.zeros(len(data))
    distances = cdist(data[idx].reshape(1, -1), data, metric='euclidean')
    points = np.argwhere(distances <= r)[:, 1]
    peak = np.mean(data[points[:]], axis=0)
    search_path_distance = cdist(peak.reshape(1, -1), data, metric='euclidean')
    search_path_points = np.argwhere(search_path_distance <= r / c)[:, 1]
    cpts[search_path_points[:]] = 1
    distance = 1000
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
    labels = np.full(len(data), -1)
    peaks = []
    number_of_labels = 0
    label_peak = {}
    peak_values = np.copy(data)
    for i, point in enumerate(tqdm.tqdm(data)):
        if labels[i] == -1:
            if path:
                peak, cpts = find_peak_opt(data, i, r, threshold, c)
                search_path = np.argwhere(cpts == 1)
                if len(peaks) != 0:
                    peakDistance = cdist(peak.reshape(1, -1), np.array(peaks))
                    points = np.argwhere(peakDistance < r / 2)[:, 1]
                    if len(points) != 0:
                        labels[i] = points[0]
                        peak_values[i] = label_peak[points[0]]
                        labels[search_path[:]] = points[0]
                    else:
                        peaks.append(peak)
                        if basin:
                            distances = cdist(peak.reshape(1, -1), data, metric='euclidean')
                            points = np.argwhere(distances <= r)[:, 1]
                            labels[points[:]] = number_of_labels
                        labels[search_path[:]] = number_of_labels
                        labels[i] = number_of_labels
                        label_peak[number_of_labels] = peak
                        peak_values[i] = peak
                        number_of_labels += 1
                else:
                    peaks.append(peak)
                    if basin:
                        distances = cdist(peak.reshape(1, -1), data, metric='euclidean')
                        points = np.argwhere(distances <= r)[:, 1]
                        labels[points[:]] = number_of_labels
                    labels[search_path[:]] = number_of_labels
                    labels[i] = number_of_labels
                    label_peak[number_of_labels] = peak
                    peak_values[i] = peak
                    number_of_labels += 1
            else:
                peak = find_peak(data, i, r)
                if len(peaks) != 0:
                    peakDistance = cdist(peak.reshape(1, -1), np.array(peaks))
                    points = np.argwhere(peakDistance < r / 2)[:, 1]
                    if len(points) != 0:
                        labels[i] = points[0]
                        peak_values[i] = label_peak[points[0]]
                    else:
                        peaks.append(peak)
                        if basin:
                            distances = cdist(peak.reshape(1, -1), data, metric='euclidean')
                            points = np.argwhere(distances <= r)[:, 1]
                            labels[points[:]] = number_of_labels
                        labels[i] = number_of_labels
                        label_peak[number_of_labels] = peak
                        peak_values[i] = peak
                        number_of_labels += 1
                else:
                    peaks.append(peak)
                    if basin:
                        distances = cdist(peak.reshape(1, -1), data, metric='euclidean')
                        points = np.argwhere(distances <= r)[:, 1]
                        labels[points[:]] = number_of_labels
                    labels[i] = number_of_labels
                    label_peak[number_of_labels] = peak
                    peak_values[i] = peak
                    number_of_labels += 1
    return labels, np.array(peaks)


def imSegment(im, r, threshold=0.01, c=4, basin_of_attraction=True, path_speedup=True, feature_type='3D'):
    imlab = rgb2lab(im)
    if feature_type == '5D':
        # if featureType is 5D --> add coordinates
        xCoords = np.expand_dims(np.tile(np.array(range(image.shape[0])).reshape(-1, 1), reps=image.shape[1]),
                                 axis=-1)
        yCoords = np.expand_dims(np.tile(np.array(range(image.shape[1])).reshape(-1, 1), reps=image.shape[0]).T,
                                 axis=-1)
        imlab = np.append(imlab, xCoords, axis=-1)
        imlab = np.append(imlab, yCoords, axis=-1)
        flattened_image = imlab.reshape(-1, 5)
    else:
        flattened_image = imlab.reshape(-1, 3)
    if basin_of_attraction or path_speedup:
        labels, peaks = mean_shift_opt(flattened_image, r, threshold, c, basin_of_attraction, path_speedup)
    else:
        labels, peaks = mean_shift(flattened_image, r)
    segemented_values = np.zeros((len(labels), 3))
    peaks = peaks[:, 0:3]
    for idx, x in enumerate(peaks):
        points = np.argwhere(labels == idx)[:, 0]
        segemented_values[points[:]] = x
    print(f'number of unique labels {np.unique(labels)}')
    print(f'number of peaks {len(peaks)}')
    segmented_image = segemented_values.reshape(im.shape)
    segmented_image_rgb = lab2rgb(segmented_image)
    return segmented_image_rgb, labels


if __name__ == '__main__':
    # data = scipy.io.loadmat('./Data/pts.mat')['data']
    # labels = np.arange(2000)
    # peaks = data
    # plotclusters3D(data.T, labels, peaks.T, 'Images/database.png')
    # labels, peaks = mean_shift(data.T, 2)
    # print(np.unique(np.array(labels)))
    # plotclusters3D(data.T, labels, np.array(peaks), 'Images/databaseClustered.png')
    radiuses = [20]
    thresholds = [0.1, 0.5]
    circles = [4]
    pictures = ["368078"]
    print("threshold changes")
    for r in radiuses:
        for t in thresholds:
            for c in circles:
                for picture in pictures:
                    image = plt.imread(f"./Data/{picture}.jpg")
                    plt.figure(figsize=(15, 4.8))
                    plt.subplot(161)
                    plt.title("Original image")
                    plt.imshow(image)
                    plt.axis('off')

                    start_time = time()
                    segmented_image_3d, labels_3d = imSegment(image, r, threshold=t, c=c, basin_of_attraction=True,
                                                              path_speedup=False)
                    print(f"Took {time() - start_time} seconds for 3D")
                    plt.subplot(162)
                    plt.title(f"Speedup 1")
                    plt.imshow(segmented_image_3d)
                    plt.axis('off')

                    start_time = time()
                    segmented_image_3d, labels_3d = imSegment(image, r, threshold=t, c=c, basin_of_attraction=False,
                                                              path_speedup=True)
                    print(f"Took {time() - start_time} seconds for 3D")
                    plt.subplot(163)
                    plt.title(f"Speedup 2")
                    plt.imshow(segmented_image_3d)
                    plt.axis('off')

                    start_time = time()
                    segmented_image_3d, labels_3d = imSegment(image, r, threshold=t, c=c, basin_of_attraction=True,
                                                              path_speedup=True)
                    print(f"Took {time() - start_time} seconds for 3D")
                    plt.subplot(164)
                    plt.title(f"Speedup 1 & 2")
                    plt.imshow(segmented_image_3d)
                    plt.axis('off')

                    # start_time = time()
                    # segmented_image_5d, labels_5d = imSegment(image, r, threshold=t, c=c, basin_of_attraction=True,
                    #                                           path_speedup=True,
                    #                                           feature_type='5D'
                    #                                           )
                    # print(f"Took {time() - start_time} seconds for 5D")
                    # plt.subplot(185)
                    # plt.title(f"5D Image")
                    # plt.imshow(segmented_image_5d)
                    # plt.axis('off')

                    equalized_image = histogram(picture)

                    plt.subplot(165)
                    plt.title("Histogram")
                    plt.imshow(equalized_image)
                    plt.axis('off')

                    start_time = time()
                    segmented_image_3d_histogram, labels_3d = imSegment(equalized_image, r, threshold=t, c=c,
                                                                        basin_of_attraction=True,
                                                                        path_speedup=True)
                    print(f"Took {time() - start_time} seconds for 3D")
                    plt.subplot(166)
                    plt.title(f"Histogram S1&2")
                    plt.imshow(segmented_image_3d_histogram)
                    plt.axis('off')

                    equalized_image = histogram(picture)

                    # start_time = time()
                    # segmented_image_5d_histogram, labels_5d = imSegment(equalized_image, r, threshold=t, c=c,
                    #                                                     basin_of_attraction=True,
                    #                                                     path_speedup=True,
                    #                                                     feature_type='5D')
                    # print(f"Took {time() - start_time} seconds for 5D")
                    # plt.subplot(188)
                    # plt.title(f"Histogram 5D ")
                    # plt.imshow(segmented_image_5d_histogram)
                    plt.axis('off')
                    plt.suptitle(f"Mean-shift 3D with r={r}, t={t}, c={c}", fontsize=14)
                    plt.savefig(f'Images/{picture} Mean-shift 3D with r={r}, t={t}, c={c}, no 5D.png')
                    # plt.show()

    # radiuses = [20, 10, 5]
    # thresholds = [0.01]
    # circles = [4]
    # pictures = ["181091"]
    # print("radius 5D")
    # for r in radiuses:
    #     for t in thresholds:
    #         for c in circles:
    #             for picture in pictures:
    #                 image = plt.imread(f"./Data/{picture}.jpg")
    #                 plt.figure(figsize=(15, 4.8))
    #                 plt.subplot(141)
    #                 plt.title("Original image")
    #                 plt.imshow(image)
    #                 plt.axis('off')
    #
    #                 start_time = time()
    #                 segmented_image_5d, labels_5d = imSegment(image, r, threshold=t, c=c, basin_of_attraction=True,
    #                                                           path_speedup=True,
    #                                                           feature_type='5D'
    #                                                           )
    #                 print(f"Took {time() - start_time} seconds for 5D")
    #                 plt.subplot(142)
    #                 plt.title(f"5D Image")
    #                 plt.imshow(segmented_image_5d)
    #                 plt.axis('off')
    #
    #                 equalized_image = histogram(picture)
    #
    #                 plt.subplot(143)
    #                 plt.title("Histogram")
    #                 plt.imshow(equalized_image)
    #                 plt.axis('off')
    #
    #                 start_time = time()
    #                 segmented_image_5d_histogram, labels_5d = imSegment(equalized_image, r, threshold=t, c=c,
    #                                                                     basin_of_attraction=True,
    #                                                                     path_speedup=True,
    #                                                                     feature_type='5D')
    #                 print(f"Took {time() - start_time} seconds for 5D")
    #                 plt.subplot(144)
    #                 plt.title(f"Histogram 5D ")
    #                 plt.imshow(segmented_image_5d_histogram)
    #                 plt.axis('off')
    #                 plt.suptitle(f"Mean-shift 5D with r={r}, t={t}, c={c}", fontsize=14)
    #                 plt.savefig(f'Images/{picture} Mean-shift 3D with r={r}, t={t}, c={c}, 5D.png')
    #                 # plt.show()