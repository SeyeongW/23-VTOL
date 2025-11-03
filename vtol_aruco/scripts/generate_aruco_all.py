#!/usr/bin/env python3
import cv2
import os

# choose dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

out_dir = "aruco_imgs"
os.makedirs(out_dir, exist_ok=True)

for marker_id in range(50):  # 0 ~ 49
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 400)
    filename = os.path.join(out_dir, f"aruco_{marker_id}.png")
    cv2.imwrite(filename, img)
    print("saved", filename)
