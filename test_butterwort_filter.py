import numpy as np
import matplotlib.pyplot as plt
import argparse
import read_input as ri
import tools as t


if __name__ == "__main__":
    dictionary_path = ''

    # load whole take file with start/end frame without T-poses
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='fname', type=int)
    args = parser.parse_args()
    fname = args.fname
    # fname = 3

    title, start_frame, end_frame, new_data, marker_list, fps, total_frames, hand = ri.read_input(fname)

    r_velocity, r_vel = t.hand_velocity(start_frame, end_frame, new_data, marker_list, 'R')

    r_acc = t.hand_acceleration(r_vel)
    cutoff = 12

    r_acc_filt = t.butter_filter(r_acc, cutoff, fps, 10)
    # r_acc_filt2 = t.butter_filter(r_acc, 2, fps, 10)
    # r_acc_filt4 = t.butter_filter(r_acc, 4, fps, 10)
    # r_acc_filt10 = t.butter_filter(r_acc, 10, fps, 10)
    zero_crossing = t.zero_crossing(r_acc_filt)

    x = np.arange(start_frame, end_frame)

    fig2 = plt.figure("{}-hand_acc-coff-{}".format(title, cutoff), figsize=(10.5, 7))

    fig2.suptitle("Hand acceleration")
    plt.plot(x, r_acc, 'c', label='Acceleration ')

    # plt.plot(x, r_acc_filt2, 'y', label='Filtered acceleration / coff = 2 ')

    # plt.plot(x, r_acc_filt4, 'g', label='Filtered acceleration / coff = 4 ')

    # plt.plot(x, r_acc_filt10, 'b', label='Filtered acceleration / coff = 10 ')

    plt.plot(x, r_acc_filt, 'm', label='Filtered acceleration / coff = {}'.format(cutoff))

    plt.plot(x[zero_crossing], r_acc_filt[zero_crossing], 'o', label='Zero-crossing points')
    plt.ylabel("Acceleration (mm/frame^2)")
    plt.xlabel("Frames")
    plt.grid(True)
    legend = fig2.legend(loc='upper right')

    plt.show()
