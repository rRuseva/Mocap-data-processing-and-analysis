import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tools as t
sys.path.insert(0, "D:\Radi\Radi RU\4ti kurs\2sm-WBU\MOCAP\Python\mocap")


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

    start_frame = 829
    end_frame = 923
    data, marker_list, fps, tot_frames = t.read_frames(filename)

    # change origin point to be between the hip's markers // True is to find the relative coordinates
    new_origin = ['RFWT', 'LFWT', 'RBWT']
    new_data = t.change_origin_point(data, new_origin, marker_list, True)

    r_loc, ch_c_r, r_reg_un = t.hand_location(start_frame, end_frame, data, marker_list, h='R')
    l_loc, ch_c_l, l_reg_un = t.hand_location(start_frame, end_frame, data, marker_list, h='L')

    print("- Right hand changes in location: {}".format(ch_c_r))
    print("- Right hand is in:")
    for reg in r_reg_un:
        print("  - region {} for {} frames ".format(reg[0], reg[1]))

    print("\n- Left hand changes in location: {}".format(ch_c_l))
    print("- Left hand is in:")
    for reg in l_reg_un:
        print("  - region {} for {} frames ".format(reg[0], reg[1]))

    x = np.arange(start_frame, end_frame)

    fig = plt.figure("{}-HandsLocation-{}-{}".format(title, start_frame, end_frame), figsize=(10.5, 7))
    fig.suptitle("Hands location for sign between {} and {} frame".format(start_frame, end_frame))

    plt.subplot(2, 1, 1)
    plt.plot(x, r_loc[:, [1]], 'r', label='Right hand')
    plt.plot(x, l_loc[:, [1]], 'g', label='Left hand')
    plt.grid(True)
    plt.xlabel("Frames")
    plt.ylabel("Regions")

    plt.subplot(2, 2, 3)
    plt.title("Right hand location changes: {}".format(ch_c_r))
    plt.hist(r_loc[:, [1]], bins=range(15), facecolor='r', align="left")
    plt.xticks(np.arange(1, 16, step=1))
    plt.ylabel("Number of Frames")
    plt.xlabel("Regions")

    plt.subplot(2, 2, 4)
    plt.title("Left hand location changes: {}".format(ch_c_l))
    plt.hist(l_loc[:, [1]], bins=range(15), facecolor='g', align="left")
    plt.xticks(np.arange(1, 16, step=1))
    plt.ylabel("Number of Frames")
    plt.xlabel("Regions")

    legend = fig.legend(loc='upper right')
    plt.show()
