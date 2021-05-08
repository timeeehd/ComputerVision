from scipy.spatial.distance import cdist
import scipy.io
import tqdm

from skimage.color import lab2rgb, rgb2lab

from plotclusters3D import *

from time import time

# def distance(point1, point2):
#     distance =

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


        peakMovements = np.sum(np.abs(old_peak_values- peak_values))
        print(f"Run {runs} with peakmovements: {peakMovements}")
        if peakMovements < t:
            optimal = True
    return labels, peaks


def find_peak(data, idx, r):
    distances = cdist(data[idx].reshape(1, -1), data, metric='euclidean')
    points = np.argwhere(distances <= r)[:, 1]
    peak = np.mean(data[points[:]], axis=0)
    distance = 1000
    while distance > 0.01:
        old_peak = peak
        distances = cdist(old_peak.reshape(1,-1), data, metric='euclidean')
        points = np.argwhere(distances <= r)[:, 1]
        peak = np.mean(data[points[:]], axis=0)
        distance = np.linalg.norm(old_peak-peak)

    return peak

def mean_shift(data, r):
    labels = np.zeros(len(data))
    peaks = []
    number_of_labels = 0
    label_peak = {}
    peak_values = np.copy(data)
    for i, point in enumerate(tqdm.tqdm(data)):
        peak = find_peak(data, i , r)
        if len(peaks) != 0:
            peakDistance = cdist(peak.reshape(1, -1), np.array(peaks))
            points = np.argwhere(peakDistance < r / 2)[:,1]
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
    return labels, np.array(peaks), peak_values




def imSegment(im, r, basin_of_attraction=True):
    imlab = rgb2lab(im)
    imlab = imlab.reshape(-1, 3)
    segemented_values2 = np.zeros(len(imlab))
    labels, peaks, segemented_values = mean_shift(imlab, r)
    segemented_values2 = np.zeros((len(labels), 3))
    for idx,x in enumerate(peaks):
        points = np.argwhere(labels == idx)[:,0]
        segemented_values2[points[:]] = x
    segmented_image = segemented_values.reshape(im.shape)
    segmented_image2 = segemented_values2.reshape(im.shape)
    # print(segmented_image == segmented_image2)
    segmented_image_rgb = lab2rgb(segmented_image)
    segmented_image2_rgb = lab2rgb(segmented_image2)
    return segmented_image_rgb, labels, segmented_image2_rgb


if __name__ == '__main__':
    # data = scipy.io.loadmat('./Data/pts.mat')['data']
    # labels = np.arange(2000)
    # peaks = data
    # plotclusters3D(data.T, labels, peaks.T, 'Images/database.png')
    # labels, peaks = mean_shift(data.T, 2)
    # print(np.unique(np.array(labels)))
    # plotclusters3D(data.T, labels, np.array(peaks), 'Images/databaseClustered.png')

    picture = "bulbasaur"
    image = plt.imread(f"./Data/{picture}.jpg")


    r = 50
    start_time = time()
    segmented_image_3d, labels_3d, segmented_image2_3d = imSegment(image, r, True)
    print(f"Took {time() - start_time} seconds for 3D")
    plt.title(f"Mean-shift 3D with r={r}")
    plt.imshow(segmented_image_3d)
    plt.savefig(f'Images/{picture} Mean-shift 3D with r={r}')
    plt.show()
    plt.title(f"Mean-shift 3D with r={r}")
    plt.imshow(segmented_image2_3d)
    plt.savefig(f'Images/{picture} Mean-shift 3D with r={r} 2')
    plt.show()
    # r = 20
    # start_time = time()
    # segmented_image_3d, labels_3d = imSegment(image, r, False)
    # print(f"Took {time() - start_time} seconds for 3D")
    # plt.title(f"Mean-shift 3D with r={r}")
    # plt.imshow(segmented_image_3d)
    # plt.savefig(f'Images/{picture} Mean-shift 3D with r={r} 2')
    # plt.show()
