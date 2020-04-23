import os
import numpy as np
import cv2

from matplotlib import pyplot as plt

root = os.getcwd()
data = os.path.join(root, 'data')

img = os.path.join(data, 'image.jpg')

CHANNEL_NUM = 3
MIN_AVERAGE_RED = 156
MAX_HUE_SHIFT = 120
THRESHOLD_RATIO = 2000

def calculate_average_color(im):
    channel_sum = np.zeros(CHANNEL_NUM)
    pixel_num = 0

    im = im/255.0

    pixel_num += (im.size/CHANNEL_NUM)
    channel_sum += np.sum(im, axis=(0, 1))

    bgr_mean = channel_sum / pixel_num

    rgb_mean = list(bgr_mean)[::-1]

    return np.matrix([[rgb_mean[0]],[rgb_mean[1]],[rgb_mean[2]]])

def calculate_standard_deviation(im):
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)
    pixel_num = 0

    im = im/255.0

    pixel_num += (im.size/CHANNEL_NUM)

    channel_sum_squared += np.sum(np.square(im), axis=(0, 1))
    bgr_mean = channel_sum / pixel_num

    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    rgb_std = list(bgr_std)[::-1]

    return rgb_std

def hue_shift_red(rgb_vector, h):
    U = np.cos(h * np.pi / 180)
    W = np.sin(h * np.pi / 180)

    hue_vector = np.matrix([[1], [U], [W]])
    # rgb_vector = np.matrix([[r], [g], [b]])

    params_matrix = np.matrix('0.299 0.701 0.168; 0.587 -0.587 0.330; 0.114 -0.144 -0.497')

    params_hue_vector = np.matmul(params_matrix, hue_vector)

    # A[0,:][:,0]

    # a = params_hue_vector[2, :][0, :]

    rgb_vector = np.multiply(params_hue_vector, rgb_vector)

    return rgb_vector

def calculate_hue_shift(average_vector):

    new_average_red = average_vector[0, :][0, :].item()
    hue_shift = 0

    # print(new_average_red)
    while new_average_red < MIN_AVERAGE_RED:
        shifted_rgb_vector = hue_shift_red(average_vector, hue_shift)
        hue_shift += 1
        new_average_red = np.sum(shifted_rgb_vector)

        # add max iteration check

        # print(new_average_red)

    return hue_shift

def create_histogram():

    return 0

def prepare_normalizing_hists(num_pixels, hists):
    threshold_level = num_pixels / THRESHOLD_RATIO


    norm_hist = []
    norm_hist_r = []
    norm_hist_g = []
    norm_hist_b = []


    norm_hist_r.append(0)
    norm_hist_g.append(0)
    norm_hist_b.append(0)

    for i in range(0, 256):
        if hists[0][i] - threshold_level < 2:
            norm_hist_r.append(i)
        if hists[1][i] - threshold_level < 2:
            norm_hist_g.append(i)
        if hists[2][i] - threshold_level < 2:
            norm_hist_b.append(i)

    norm_hist_r.append(255)
    norm_hist_g.append(255)
    norm_hist_b.append(255)

    norm_hist.append(norm_hist_r)
    norm_hist.append(norm_hist_g)
    norm_hist.append(norm_hist_b)

    return norm_hist

def normalizing_interval(hist):
    low = 0
    high = 255
    max_distance = 0

    for i in range(1, len(hist)):
        distance = hist[i] - hist[i-1]
        if distance > max_distance:
            max_distance = distance
            high = hist[i]
            low = hist[i-1]

    return low, high

def main():
    im = cv2.imread(img) # image in M*N*CHANNEL_NUM shape, channel in BGR order
    rows, cols, channels = im.shape
    num_pixels = rows * cols

    # memi = None
    #
    # for i, mem in enumerate(im):
    #     if i == 2:
    #         break
    #     # print(mem)
    #     # print(type(mem))
    #     # print('\n')
    #     memi = mem
    #
    # for j in memi:
    #     print(j)
    #
    # return 0
    hb, hg, hr = None, None, None

    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([im], [i], None, [256], [0, 256])
        if col == 'b':
            hb = hist
        elif col == 'g':
            hg = hist
        else:
            hr = hist

        for i, j in enumerate(hist):
            print(i, j)

        plt.plot(hist,color = col)
        plt.xlim([0,256])

        # print(type(hist))
    plt.show()

    hists = []
    hists.append(hr)
    hists.append(hg)
    hists.append(hb)

    norm_hists = prepare_normalizing_hists(rows*cols, hists)

    low, high = normalizing_interval(norm_hists[0])

    print(low, high)

    # print(hb, hg, hr)

    average_vector = calculate_average_color(im)


    hue_shift = calculate_hue_shift(average_vector * 255)

    # rgb_shift_vector = hue_shift_red(average_vector, 0)


    # print(average_vector, rgb_shift_vector)
    return 0


main()
