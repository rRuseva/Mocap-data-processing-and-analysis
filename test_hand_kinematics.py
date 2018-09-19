import numpy as np
import matplotlib.pyplot as plt
import argparse
import read_input as ri
import tools as t

if __name__ == "__main__":
    # load whole take file with start/end frame without T-poses
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='fname', type=int)
    args = parser.parse_args()
    fname = args.fname
    # fname = 3

    title, start_frame, end_frame, new_data, marker_list, fps, total_frames, hand = ri.read_input(fname)

    r_hand_tr, r_arm_tr = t.hand_trajectory(start_frame, end_frame, new_data, marker_list, 'R')
    r_velocity, r_vel = t.hand_velocity(start_frame, end_frame, new_data, marker_list, 'R')

    r_acc = t.hand_acceleration(r_vel)
    r_acc_filt = t.butter_filter(r_acc, 12, fps, 10)
    # zero_crossing = t.zero_crossing(r_acc_filt)

    x = np.arange(start_frame, end_frame)

    # fig = plt.figure("{}-kinematics-{}-{}".format(title, start_frame, end_frame), figsize=(10.5,7))
    # fig.suptitle("Right hand kinematics for sign between {} and {} frame".format(start_frame, end_frame))

    # plt.subplot(3, 1, 1)
    fig1 = plt.figure("{}-hand-trajectory-{}-{}".format(title, start_frame, end_frame), figsize=(10.5, 7))
    plt.plot(x, r_hand_tr[:, [0]], 'r', label='x')
    plt.plot(x, r_hand_tr[:, [1]], 'g', label='y')
    plt.plot(x, r_hand_tr[:, [2]], 'b', label='z')
    plt.ylabel("Trajectory (mm)")
    plt.xlabel("Frames")
    plt.grid(True)
    legend = fig1.legend(loc='upper right')

    fig2 = plt.figure("{}-hand-velocity-{}-{}".format(title, start_frame, end_frame), figsize=(10.5, 7))
    plt.plot(x, r_velocity[:, [0]], 'r', label='x')
    plt.plot(x, r_velocity[:, [1]], 'g', label='y')
    plt.plot(x, r_velocity[:, [2]], 'b', label='z')
    plt.plot(x, r_vel, 'm', label='Normalized velocity')
    plt.ylabel("Velocity (mm/frame)")
    plt.xlabel("Frames")
    plt.grid(True)

    legend = fig2.legend(loc='upper right')

    fig3 = plt.figure("{}-hand-acceleration-{}-{}".format(title, start_frame, end_frame), figsize=(10.5, 7))
    plt.plot(x, r_acc, 'c', label="Acceleration")
    plt.plot(x, r_acc_filt, 'm', label='Filtered acceleration')
    plt.ylabel("Acceleration (mm/frame^2)")
    plt.xlabel("Frames")
    plt.grid(True)

    legend = fig3.legend(loc='upper right')

    plt.show()
