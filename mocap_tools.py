import c3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz


def marker_list_read(_file):
    _marker_list = []
    with open(_file, 'r') as w:
        for line in w:
            _marker_list.append(line.split('\n')[0])

    return _marker_list


def marker_list_save(_marker_list, _file):
    with open(_file, 'r') as w:
        for item in _marker_list:
            w.write(item + '\n')


def compare_markers(mlist, _marker_list):
    indexes = np.zeros((len(mlist)))
    for n, m in enumerate(mlist):
        for i in range(len(_marker_list)):
            if m in _marker_list[i]:
                # print(m + ' : ' + str(i) + ' (' + _marker_list[i] + ')')
                indexes[n] = int(i)
    return indexes


def c3d_load(_file):
    """ loads c3d file

    :param _file: path
    :return:
        trajectory: ndarray (marker, frame, channel),
        marker_list: list (of markers - same order as trajectory),
        _fps: framerate of data
    """

    with open(_file, 'rb') as _handle:
        _reader = c3d.Reader(_handle)
        _fps = int(_reader.header.frame_rate)
        _tot_markers = _reader.point_used   # unlabeled markers included
        _tot_frames = _reader.last_frame() - _reader.first_frame()+1  # first frame number = 1
        _trajectory = np.zeros((_tot_markers, _tot_frames, 5))  # [marker, frame, channel]
        for _i, (_points) in enumerate(_reader.read_frames()):
            # print('Frame {}: {}'.format(i, points))
            for _m in range(_tot_markers):
                _trajectory[_m, _i, :] = _points[1][_m]
        _marker_list = []
        for _i in range(_tot_markers):
            _marker_list.append(_reader.point_labels[_i].strip())
    return _trajectory, _marker_list, _fps


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def extraction_refinement(full_trajectory, SMI, interval_of_interest, speed_threshold=0.9, frames_below_thr=30, visualize=False, version=2):
    """ refinement based on speed of 1 marker

    :param speed_threshold:
    :param frames_below_thr:
    :param dict_trajectory:
    :param SMI:
    :param raw_interval:
    :param visualize:
    :return:
    """
    # TODO: refinment for multiple markers

    trajectory = full_trajectory[interval_of_interest[0]:interval_of_interest[1], 3*SMI:3*(SMI+1)]
    speed = np.diff(trajectory, axis=0)
    # norm_speed = np.sqrt(np.square(speed[:, 0]) + np.square(speed[:, 1]) + np.square(speed[:, 2]))
    norm_speed = np.linalg.norm(speed, axis=1)
    norm_acc = np.diff(norm_speed, axis=0)
    # plt.plot(trajectory)
    # plt.plot(norm_speed)
    plt.axhline(y=speed_threshold)
    # plt.show()
    # -------------------------------------------------------
    # MOVE DETECTION
    below = np.zeros(norm_speed.shape[0])
    below_marks = []
    start_mark = 0
    for i in range(norm_speed.shape[0]-1):
        if norm_speed[i] < speed_threshold:
            below[i+1] = below[i] + 1
        if below[i+1] == 1:
            start_mark = i+1
        if below[i] != 0 and below[i+1] == 0:
            if below[i] > frames_below_thr:
                below_marks.append([start_mark, i])
        if below[i] != 0 and (norm_speed.shape[0]-2) == i:
            if below[i] > frames_below_thr:
                below_marks.append([start_mark, i])
    print(below_marks)
    move_detection = [below_marks[0][1], below_marks[1][0]]
    POI = [[move_detection[0], move_detection[1]]]
    # -------------------------------------------------------
    # FILTRATION
    order = 3
    fs = 120.0       # sample rate, Hz
    cutoff = 15  # desired cutoff frequency of the filter, Hz
    acc_filtered = butter_lowpass_filter(norm_acc, cutoff, fs, order)

    # -------------------------------------------------------
    # ACCELERATION ANALYSIS
    acc_filtered_range = acc_filtered[move_detection[0]:move_detection[1]]
    zero_crossings = np.where(np.diff(np.sign(acc_filtered_range)))[0]
    POI.append([zero_crossings[0] + move_detection[0], zero_crossings[-1] + move_detection[0]])
    POI.append([zero_crossings[1] + move_detection[0], zero_crossings[-2] + move_detection[0]])
    minmax = np.where(np.diff(np.sign(np.diff(acc_filtered_range))))[0]

    first = True
    last = True
    for i in range(len(minmax)):
        if minmax[i] + move_detection[0] + 1 > POI[1][0] and first:
            first = False
            first_value = minmax[i] + move_detection[0] + 1
        if minmax[i] + move_detection[0] + 1 > POI[1][1] and last:
            last = False
            last_value = minmax[i-1] + move_detection[0] - 1

    POI.append([first_value, last_value])
    # print(POI)
    if visualize:
        plt.plot(norm_speed, 'r')
        jet = plt.get_cmap('rainbow')
        colors = iter(jet(np.linspace(0, 1, 4)))
        colors2 = iter(jet(np.linspace(0, 1, 4)))
        for i in range(len(POI)):
            plt.axvline(x=POI[i][0], color=next(colors))
            plt.axvline(x=POI[i][1], color=next(colors2))
            # next(colors)
        plt.show()
    return POI


def dictionary_marker_reduction(sign, full_marker_list, reduced_marker_list, zero_by=None):
    SMI_D = compare_markers(reduced_marker_list, full_marker_list)
    zero_by = full_marker_list[int(compare_markers([zero_by], full_marker_list)[0])]
    sign_reduced = np.zeros((len(SMI_D), sign.shape[1], 3))  # reduced no of markers and channels
    SMI_zero = full_marker_list.index(zero_by)
    for i, s in enumerate(SMI_D):
        if int(s) == SMI_zero:
            SMI_zero_idx = i
        sign_reduced[i, :, :] = sign[int(s), :, :3] - sign[SMI_zero, :, :3]
        # sign_reduced[i, :, :] = sign[SMI_zero, :, :3] - sign[int(s), :, :3]
    sign_reduced = np.delete(sign_reduced, SMI_zero_idx, 0)
    return sign_reduced


def dimension_reduction(data):
    result = np.zeros((data.shape[1], data.shape[0]*data.shape[2]))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            result[:, 3*i+j] = data[i, :, j]
    return result


def rotate_data(data, rotation=np.array([[2, 0, 1], [-1, -1, 1]])):
    tmp = np.zeros((int(data.shape[1]/3), data.shape[0], 3))
    for i in range(data.shape[1]):
        if (i % 3) == 0:
            tmp[int(i / 3), :, rotation[0, 0]] = rotation[1, 0] * (data[:, i])
        elif (i % 3) == 1:
            tmp[int(i / 3), :, rotation[0, 1]] = rotation[1, 1] * (data[:, i])
        elif (i % 3) == 2:
            tmp[int(i / 3), :, rotation[0, 2]] = rotation[1, 2] * (data[:, i])
    return tmp


def dictionary_separator(trajectory, speed_threshold=0.9, frames_below_thr=40):
    speed = np.diff(trajectory, axis=0)
    norm_speed = np.sqrt(np.square(speed[:, 0] + np.square(speed[:, 1]) + np.square(speed[:, 2])))
    below = np.zeros(norm_speed.shape[0])
    below_marks = []
    start_mark = 0
    for i in range(norm_speed.shape[0]-1):
        if norm_speed[i] < speed_threshold:
            below[i+1] = below[i] + 1
        if below[i+1] == 1:
            start_mark = i+1
        if below[i] != 0 and below[i+1] == 0:
            if below[i] > frames_below_thr:
                below_marks.append([start_mark, i])
        if below[i] != 0 and (norm_speed.shape[0]-2) == i:
            if below[i] > frames_below_thr:
                below_marks.append([start_mark, i])
    sep = []
    for i in range(len(below_marks)):
        sep.append(int(np.average(below_marks[i])))
    return sep


def manual_segmentation_load(file):
    with open(file) as f:
        tmp = f.readlines()
    sign_name_list = []
    sign_ends = []
    for i in range(len(tmp)):
        temp = (tmp[i].split(' '))
        sign_name_list.append(temp[0])
        sign_ends.append(int(temp[1]))

    return sign_ends, sign_name_list