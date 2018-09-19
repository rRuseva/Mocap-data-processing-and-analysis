import numpy as np
import matplotlib.pyplot as plt
import argparse
import tools as t
import read_input as ri


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
    start_frame = args.start_frame
    end_frame = args.end_frame
    # start_frame = 5285
    # end_frame = 5529

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
