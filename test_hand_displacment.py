import numpy as np
import matplotlib.pyplot as plt
import argparse
import read_input as ri
import tools as t

if __name__ == "__main__":

    # load whole take file with start/end frame without T-poses
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='fname', type=int)
    parser.add_argument(dest='start_frame', type=int)
    parser.add_argument(dest='end_frame', type=int)
    args = parser.parse_args()
    fname = args.fname
    # fname = 3

    title, start_frame, end_frame, data, marker_list, fps, total_frames, hand = ri.read_input(fname)
    # start_frame = args.start_frame
    # end_frame = args.end_frame
    start_frame = 5285
    end_frame = 5529
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
    print(message)

    tot_fr = end_frame - start_frame
    x = np.arange(start_frame, end_frame)

    fig1 = plt.figure("{}-R-Hand-displacment-{}-{}".format(title, start_frame, end_frame), figsize=(10.5, 7))
    fig1.suptitle(message)

    plt.subplot(2, 1, 1)
    plt.title("Right Hand movement")
    plt.grid(True)
    plt.plot(x, diff_right_hand[:, [0]], 'r', label='x')
    plt.plot(x, diff_right_hand[:, [1]], 'g', label='y')
    plt.plot(x, diff_right_hand[:, [2]], 'b', label='z')
    plt.xlabel("{} frames".format(tot_fr))
    plt.ylabel("Displacment")

    plt.subplot(2, 1, 2)
    plt.title("Right Arm movement")
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
    plt.title("Left Hand movement")
    plt.grid(True)
    plt.plot(x, diff_left_hand[:, [0]], 'r', label='x')
    plt.plot(x, diff_left_hand[:, [1]], 'g', label='y')
    plt.plot(x, diff_left_hand[:, [2]], 'b', label='z')
    plt.xlabel("{} frames".format(tot_fr))
    plt.ylabel("Displacment")

    plt.subplot(2, 1, 2)
    plt.title("Left Arm movement")
    plt.grid(True)
    plt.plot(x, diff_LWRE[:, [0]], 'r')
    plt.plot(x, diff_LWRE[:, [1]], 'g')
    plt.plot(x, diff_LWRE[:, [2]], 'b')
    plt.xlabel("{} frames".format(tot_fr))
    plt.ylabel("Displacment")

    legend = fig2.legend(loc='upper right')

    plt.show()
