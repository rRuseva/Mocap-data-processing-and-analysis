import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import tools as t
import read_input as ri

if __name__ == "__main__":

    # load whole take file with start/end frame without T-poses
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='fname', type=int)
    args = parser.parse_args()
    fname = args.fname
    # fname = 4

    title, start_frame, end_frame, new_data, marker_list, fps, total_frames, hand = ri.read_input(fname)

    r_velocity, r_vel = t.hand_velocity(start_frame, end_frame, new_data, marker_list, hand)
    median = np.median(r_vel)
    print("Median = ", median)
    average = np.average(r_vel)
    print("Average = ", average)
    mode = stats.mode(r_vel)
    print("Mode = ", mode)

    r_acc = t.hand_acceleration(r_vel)
    r_acc_filt = t.butter_filter(r_acc, 12, fps, 10)
    zero_crossing = t.zero_crossing(r_acc_filt)

    x = np.arange(start_frame, end_frame)

    fig1 = plt.figure("{}-vel-acc-extremums".format(title), figsize=(10.5, 7))
    fig1.suptitle("Right hand velocity and acceleration graph for sign between {} and {} frame".format(start_frame, end_frame))

    plt.subplot(2, 1, 1)
    plt.plot(x, r_vel, 'c', label='Normilized velocity')
    plt.plot(x[zero_crossing], r_vel[zero_crossing], 'o')
    plt.axhline(y=median, color='r', linestyle='-', label="Treshold - median - {}".format(median))
    plt.axhline(y=average, color='g', linestyle='-', label="Treshold - average - {}".format(average))
    plt.axhline(y=mode[0], color='b', linestyle='-', label="Treshold - mode - {}".format(mode[0]))
    plt.ylabel("Velocity (mm/frame)")
    plt.xlabel("Frames")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.plot(x, r_acc_filt, 'm', label='Filtered acceleration')
    plt.plot(x[zero_crossing], r_acc_filt[zero_crossing], 'o')
    plt.ylabel("Acceleration (mm/frame^2)")
    plt.xlabel("Frames")
    plt.grid(True)

    legend = fig1.legend(loc='upper right')
    plt.show()
