import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tools as t
sys.path.insert(0, "D:\Radi\MyProjects\MOCAP")

if __name__ == "__main__":
    dictionary_path = ''

    # filename = os.path.join(dictionary_path, 'projevy_pocasi_01_ob_rh_lh_b_g_face_gaps.c3d')
    # title = 'Pocasi_01'
    # start_frame = 600
    # end_frame = 7445

    # filename = os.path.join(dictionary_path, 'projevy_pocasi_02_ob_rh_lh_b_g_face_gaps.c3d')
    # title = 'Pocasi_02'
    # start_frame = 608
    # end_frame = 6716

    # filename = os.path.join(dictionary_path, 'projevy_pocasi_03_ob_rh_lh_b_g_face_gaps.c3d')
    # title = 'Pocasi_03'
    # start_frame = 743
    # end_frame = 6680

    filename = os.path.join(dictionary_path, 'projevy_pocasi_04_ob_rh_lh_b_g_face_gaps.c3d')
    title = 'Pocasi_04'
    start_frame = 600
    end_frame = 5115

    data, marker_list, fps, tot_frames = t.read_frames(filename)

    # change origin point to be between the hip's markers // True is to find the relative coordinates
    new_origin = ['RFWT', 'LFWT', 'RBWT']
    new_data = t.change_origin_point(data, new_origin, marker_list, True)

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
