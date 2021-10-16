import cv2
import numpy as np
from pathlib import Path


data_root = 'data/cubicasa5k'
data_file = 'train.txt'

sizes = []
for f in Path(data_root, data_file).read_text().split('\n'):
	for img_file in Path(data_root, f[1:]).iterdir():
		try:
			if '.png' == img_file.suffix and 'original' in img_file.name:
				img = cv2.imread(str(img_file), 0)
				## (1) Convert to gray, and threshold
				# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				threshed = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

				## (2) Morph-op to remove noise
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
				morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

				## (3) Find the max-area contour
				cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
				cnt = sorted(cnts, key=cv2.contourArea)[-1]

				## (4) Crop and save it
				x,y,w,h = cv2.boundingRect(cnt)
				dst = img[y:y+h, x:x+w]

				max_size = np.max(dst.shape)
				target_size = 256
				if max_size > target_size:
					imgW, imgH = dst.shape
					r = target_size/max_size
					dim = (target_size if imgW > imgH else int(r*imgW), target_size if imgH > imgW else int(r*imgH))
					dst = cv2.resize(dst, dim)

				w, h = dst.shape
				padding_width = (target_size - w)
				padding_left = int(padding_width/2)
				padding_right = padding_width - padding_left
				padding_height = (target_size - h)
				padding_top = int(padding_height/2)
				padding_bot = padding_height - padding_top

				dst = cv2.copyMakeBorder(dst, padding_left, padding_right, padding_top, padding_bot, cv2.BORDER_CONSTANT, value=[255, 255, 255])

				# Morph open to remove noise
				kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
				opening = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel, iterations=1)

				# Find contours and remove small noise
				cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				cnts = cnts[0] if len(cnts) == 2 else cnts[1]
				for c in cnts:
					area = cv2.contourArea(c)
					if area < 50:
						cv2.drawContours(opening, [c], -1, 0, -1)

				# Invert and apply slight Gaussian blur
				dst = 255 - opening
				dst = cv2.GaussianBlur(dst, (3,3), 0)

				cv2.imwrite(f"data/transformed/{Path(f).name}_{img_file.name}", dst)
		except Exception:
			pass


