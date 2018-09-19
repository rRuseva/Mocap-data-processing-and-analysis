import numpy as np
import matplotlib.pyplot as plt
import argparse
import tools as t
import read_input as ri


if __name__ == "__main__":
    # load whole take file with start/end frame without T-poses
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='fname', type=int)
    args = parser.parse_args()
    fname = args.fname
    # fname = 3

    title, start_frame, end_frame, new_data, marker_list, fps, total_frames, hand = ri.read_input(fname)

    r_velocity, r_vel = t.hand_velocity(start_frame, end_frame, new_data, marker_list, 'R')
    median = np.median(r_vel)
    print("Median = ", median)
    r_acc = t.hand_acceleration(r_vel)
    r_acc_filt = t.butter_filter(r_acc, 12, fps, 10)
    zero_crossing = t.zero_crossing(r_acc_filt)

    signs = t.segment_signs(start_frame, end_frame, new_data, marker_list, fps, median)
    count = len(signs)

    print(count)
    print()

    rest_pose_vel = np.zeros([end_frame - start_frame])

    for i in range(0, count - 1):
        st = signs[i][1]
        en = signs[i + 1][0]
        n = en - st
        vel, vel_norm = t.hand_velocity(st + start_frame, en + start_frame, new_data, marker_list, 'R')
        rest_pose_vel[st:en] = vel_norm

    tr = np.amax(rest_pose_vel)
    print("New threshold = ", tr)
    signs2 = t.segment_signs(start_frame, end_frame, new_data, marker_list, fps, tr)
    count2 = len(signs2)

    x = np.arange(start_frame, end_frame)

    fig1 = plt.figure("{}-rest-pose-vel".format(title), figsize=(10.5, 7))
    fig1.suptitle("Right hand RP velocity for sign between {} and {} frame".format(start_frame, end_frame))

    plt.plot(x, rest_pose_vel, 'c', label='Normilized velocity')
    plt.axhline(y=tr, color='m', linestyle='-', label="New Treshold")
    plt.axhline(y=median, color='r', linestyle='-', label="Treshold")
    plt.ylabel("Velocity (mm/frame)")
    plt.xlabel("Frames")
    plt.grid(True)
    legend = fig1.legend(loc='upper right')

    fig2 = plt.figure("{}-{}signs-vel".format(title, count2), figsize=(10.5, 7))
    fig1.suptitle("Right hand velocity for sign between {} and {} frame".format(start_frame, end_frame))

    plt.plot(x, r_vel, 'c', label='Normilized velocity')
    plt.plot(x[zero_crossing], r_vel[zero_crossing], 'o')
    plt.plot(x[signs[:, 0]], r_vel[signs[:, 0]], 'rs', label="Start 1 (End RP)")
    plt.plot(x[signs[:, 1]], r_vel[signs[:, 1]], 'r*', label="End 1 (Start RP")

    plt.plot(x[signs2[:, 0]], r_vel[signs2[:, 0]], 'bs', label="Start 2 (End RP)")
    plt.plot(x[signs2[:, 1]], r_vel[signs2[:, 1]], 'b*', label="End 2 (Start RP")

    plt.axhline(y=median, color='r', linestyle='-', label="Threshold = {}".format(median))

    plt.axhline(y=tr, color='m', linestyle='-', label="Threshold = {}".format(tr))

    plt.ylabel("Velocity (mm/frame)")
    plt.xlabel("Frames")
    plt.grid(True)
    legend = fig2.legend(loc='upper right')

    plt.show()
