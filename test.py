import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import argrelextrema
sys.path.insert(0, "D:\Radi\Radi RU\4ti kurs\2sm-WBU\MOCAP\Python\mocap")
# import mocap_tools

import tools as t

if __name__ == "__main__":
	dictionary_path = ''
	###
	#
	# load whole take file with start/end frame without T-poses
	#
	###

	filename = os.path.join(dictionary_path, 'projevy_pocasi_01_ob_rh_lh_b_g_face_gaps.c3d')
	title = 'Pocasi_01'
	start_frame = 600
	end_frame = 7445

	data, marker_list, fps = t.read_frames(filename)
	# print(len(marker_list))
	
	hand = t.hand_marker(data, marker_list, 'R')