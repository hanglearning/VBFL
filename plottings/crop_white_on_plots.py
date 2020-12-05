# https://www.programmersought.com/article/8966116355/
# https://stackoverflow.com/questions/59758904/check-if-image-is-all-white-pixels-with-opencv

import cv2 as io
import sys
import numpy as np

im = io.imread(sys.argv[1])

def corp_margin(img):
	img2=img.sum(axis=2)
	(row,col)=img2.shape
	row_top=0
	raw_down=0
	col_top=0
	col_down=0
	for r in range(0,row):
		if np.mean(img2[r]) != 765:
				row_top=r
				break

	for r in range(row-1,0,-1):
		if np.mean(img2[r]) != 765:
				raw_down=r
				break

	for c in range(0,col):
		if img2.sum(axis=0)[c]/row != 765:
				col_top=c
				break

	for c in range(col-1,0,-1):
		if img2.sum(axis=0)[c]/row != 765:
				col_down=c
				break

	new_img=img[row_top:raw_down+1,col_top:col_down+1,0:3]
	return new_img

img_re = corp_margin(im)
io.imwrite(f"/Volumes/BOOTCAMP/AAAI_plots/cropped/{sys.argv[1].split('/')[-1]}",img_re)
# io.imshow(img_re)