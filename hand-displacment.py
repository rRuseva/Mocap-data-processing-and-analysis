import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "D:\Radi\MyProjects\MOCAP")
# import mocap_tools

# import tools_new as t
import tools as t

if __name__ == "__main__":
    dictionary_path = ''

    filename = os.path.join(dictionary_path, 'projevy_pocasi_01_ob_rh_lh_b_g_face_gaps.c3d')
    title = 'Pocasi_01'
    start_frame = 600
    end_frame = 7445

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

    # filename = os.path.join(dictionary_path, 'projevy_pocasi_04_ob_rh_lh_b_g_face_gaps.c3d')
    # title = 'Pocasi_04'
    # start_frame = 600
    # end_frame = 5115

    data, marker_list, fps = t.read_frames(filename)

    # change origin point to be between the hip's markers // True is to find the relative coordinates
    new_origin = ['RFWT', 'LFWT', 'RBWT']
    new_data = t.change_origin_point(data, new_origin, marker_list, True)

    diff_right_hand, diff_left_hand, diff_RWRE, diff_LWRE, right_dominant, one_hand = t.hand_displacment(start_frame, end_frame, data, marker_list)
    if(right_dominant == 1):
        print("Dominant hand is: Right \n")
    else:
        print("Dominant hand is: Left hand \n")

    message = "The sign between {} - {} frames is".format(start_frame, end_frame)
    if(one_hand == 3):
        message = message + ' two handed'
    elif(one_hand == 1):
        message = message + ' right handed'
    elif(one_hand == 2):
        message = message + " left handed"
    else:
        message = message + " with no hands"

    tot_fr = end_frame - start_frame
    x = np.arange(start_frame, end_frame)

    fig1 = plt.figure("{}-R-Hand-displacment-{}-{}".format(title, start_frame, end_frame), figsize=(10.5, 7))
    fig1.suptitle(message)

    plt.subplot(2, 1, 1)
    plt.title("Hand movement")
    plt.grid(True)
    plt.plot(x, diff_right_hand[:, [0]], 'r', label='x')
    plt.plot(x, diff_right_hand[:, [1]], 'g', label='y')
    plt.plot(x, diff_right_hand[:, [2]], 'b', label='z')
    plt.xlabel("{} frames".format(tot_fr))
    plt.ylabel("Displacment")

    plt.subplot(2, 1, 2)
    plt.title("Arm movement")
    plt.grid(True)
    plt.plot(x, diff_RWRE[:, [0]], 'r')
    plt.plot(x, diff_RWRE[:, [1]], 'g')
    plt.plot(x, diff_RWRE[:, [2]], 'b')
    plt.xlabel("{} frames".format(tot_fr))
    plt.ylabel("Displacment")

    legend = fig1.legend(loc='upper right')

    fig2 = plt.figure("{}-L-Hand-displacment-{}-{}".format(title, start_frame, end_frame), figsize=(10.5, 7))
    fig2.suptitle(message)

    plt.subplot(2, 1, 1)
    plt.title("Hand movement")
    plt.grid(True)
    plt.plot(x, diff_left_hand[:, [0]], 'r', label='x')
    plt.plot(x, diff_left_hand[:, [1]], 'g', label='y')
    plt.plot(x, diff_left_hand[:, [2]], 'b', label='z')
    plt.xlabel("{} frames".format(tot_fr))
    plt.ylabel("Displacment")

    plt.subplot(2, 1, 2)
    plt.title("Arm movement")
    plt.grid(True)
    plt.plot(x, diff_LWRE[:, [0]], 'r')
    plt.plot(x, diff_LWRE[:, [1]], 'g')
    plt.plot(x, diff_LWRE[:, [2]], 'b')
    plt.xlabel("{} frames".format(tot_fr))
    plt.ylabel("Displacment")

    legend = fig2.legend(loc='upper right')

    plt.show()
