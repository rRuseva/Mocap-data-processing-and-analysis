import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "D:\Radi\MyProjects\MOCAP")
# import mocap_tools

import tools as t

if __name__ == "__main__":
    dictionary_path = ''

    # filename = os.path.join(dictionary_path, 'projevy_pocasi_01_ob_rh_lh_b_g_face_gaps.c3d')
    # title = 'Pocasi_01'
    # start_frame = 600
    # end_frame = 7445

    # start_frame = 1000
    # end_frame = 1710

    # filename = os.path.join(dictionary_path, 'projevy_pocasi_02_ob_rh_lh_b_g_face_gaps.c3d')
    # title = 'Pocasi_02'
    # start_frame = 608
    # end_frame = 6716

    # #end_frame = 1200
    # start_frame = 2500
    # end_frame = 3500

    # filename = os.path.join(dictionary_path, 'projevy_pocasi_03_ob_rh_lh_b_g_face_gaps.c3d')
    # title = 'Pocasi_03'
    # start_frame = 743
    # end_frame = 6680

    filename = os.path.join(dictionary_path, 'projevy_pocasi_04_ob_rh_lh_b_g_face_gaps.c3d')
    title = 'Pocasi_04'
    start_frame = 600
    end_frame = 5115

    data, marker_list, fps = t.read_frames(filename)

    # change origin point to be between the hip's markers // True is to find the relative coordinates
    new_origin = ['RFWT', 'LFWT', 'RBWT']
    new_data = t.change_origin_point(data, new_origin, marker_list, True)

    r_velocity, r_vel = t.hand_velocity(start_frame, end_frame, new_data, marker_list, 'R')
    median = np.median(r_vel)
    print("Median=", median)
    r_acc = t.hand_acceleration(r_vel)
    r_acc_filt = t.butter_filter(r_acc, 12, fps, 10)
    zero_crossing = t.zero_crossing(r_acc_filt)

    signs = t.segment_signs(start_frame, end_frame, new_data, marker_list, fps, median)
    count = len(signs)

    print(count)
    print()

    rest_pose_vel = np.zeros([end_frame - start_frame])

    for i in range(0, count - 1):
        # print(i)
        st = signs[i][1]
        en = signs[i + 1][0]
        n = en - st
        # print(st, en)
        # print(n)

        # print(new_data[st:en, :, :])
        vel, vel_norm = t.hand_velocity(st + start_frame, en + start_frame, new_data, marker_list, 'R')
        # print(vel_norm.shape)
        # print(rest_pose_vel[st:en].shape)
        # rest_pose_vel[st:en] = vel_norm

        rest_pose_vel[st:en] = vel_norm

        # print()

    # rest_pose_vel = np.array(rest_pose_vel)
    # print(rest_pose_vel)

    tr = np.amax(rest_pose_vel)
    print("threshold=", tr)
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
    plt.plot(x[signs[:, 0]], r_vel[signs[:, 0]], 'rs', label="Start 1 ( end RP)")
    plt.plot(x[signs[:, 1]], r_vel[signs[:, 1]], 'r*', label="End 1 (Start RP")

    plt.plot(x[signs1[:, 0]], r_vel[signs1[:, 0]], 'gs', label="Start 1")
    plt.plot(x[signs1[:, 1]], r_vel[signs1[:, 1]], 'g*', label="End 1")

    plt.plot(x[signs2[:, 0]], r_vel[signs2[:, 0]], 'bs', label="Start 2 ( end RP)")
    plt.plot(x[signs2[:, 1]], r_vel[signs2[:, 1]], 'b*', label="End 2 (Start RP")

    # plt.plot(x[signs3[:, 0]], r_vel[signs3[:, 0]], 'ms', label = "Start 2")
    # plt.plot(x[signs3[:, 1]], r_vel[signs3[:, 1]], 'm*', label = "End 2")

    plt.axhline(y=median, color='r', linestyle='-', label="Treshold")

    plt.axhline(y=tr, color='m', linestyle='-', label="Treshold")

    plt.ylabel("Velocity (mm/frame)")
    plt.xlabel("Frames")
    plt.grid(True)
    legend = fig2.legend(loc='upper right')

    plt.show()
