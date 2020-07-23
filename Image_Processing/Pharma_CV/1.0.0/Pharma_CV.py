__author__ = "Rafael Lopes Almeida"
__maintainer__ = "Rafael Lopes Almeida"
__email__ = "fael.rlopes@gmail.com"

__date__ = "21/07/2020"
__version__ = "1.0.0"

# ------------------------------------------------------------------------------------------- Chamando libs
import utils
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
import platform
import datetime
import time
start_time = time.time()

system_os = platform.system()

# ------------------------------------------------------------------------------------------- Parametros
N_BLISTER = 3
N_PILLS = 8

AREA_THRESHOLD = 0
MEAN_AREA = 0

# ------------------------------------------------------------------------------------------- MAIN

orig_stdout = sys.stdout

if system_os == 'Windows':
    b = open('D:/Code/Source/Pharma CV/log/log_terminal.txt', 'a')
elif system_os == 'Linux':
    b = open('D:/Code/Source/Pharma CV/log/log_terminal.txt', 'a')
sys.stdout = b

for current_iter in range(1, 2):
    # ------------------------------------------------------------------------------------------- Imagem
    # -------------------------------------------- Abrir imagem
    try:
        if system_os == 'Windows':
            frame = cv2.imread(
                f'D:/Code/Source/Pharma CV/samples/neralgyn/{current_iter}.png', 1)
                
        elif system_os == 'Linux':
            frame = cv2.imread(
                f'D:/Code/Source/Pharma CV/samples/neralgyn/{current_iter}.png', 1)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        print('Erro: Carregar imagem')

    # -------------------------------------------- Camera input
    # try:
    #     frame_origin = camera_capture(0)
    # except:
    #     print('Erro: Captura da camera')

    # frame = frame_origin[180:400, 120:500]

    now = datetime.datetime.now()
    # date_stamp = (f'{now.year}.{now.month}.{now.day}._{now.hour}h{now.minute}m{now.second}s')
    date_stamp = (f'{current_iter}')

    # utils.show_img(frame, 'Imagem')
    # utils.save_img(frame, 'Cropped')

    # ------------------------------------------------------------------------------------------- Gamma
    frame_gamma = utils.adjust_gamma(frame, gamma=2.0)

    # utils.show_img(frame_gamma, 'Gamma')
    # utils.save_img(frame, 'Gamma', date_stamp)

    # ------------------------------------------------------------------------------------------- CLACHE
    frame_clache = utils.apply_clache(
        frame_gamma, clip_Limit=2.0, GridSize=8)

    # utils.show_img(frame_clache, 'Clache')
    # utils.save_img(frame, 'Clache', date_stamp)

    # ------------------------------------------------------------------------------------------- K-MEAN
    frame_blur = cv2.medianBlur(frame_clache,5)

    frame_mean, colors_mean_cv, colors_mean = utils.k_mean(
        frame_blur, K_iter=5, criteria_iter=50, criteria_eps=50)

    # utils.show_img(frame_mean, 'K Means')
    # utils.save_img(frame_mean, 'K-Mean', date_stamp)c

    # ------------------------------------------------------------------------------------------- Filter HSV
    frame_hsv = cv2.cvtColor(frame_mean, cv2.COLOR_BGR2HSV)

    frame_bin = np.zeros(frame_mean.shape[:2], np.uint32)
    # plt.imshow(frame_bin, cmap='gray')
    # plt.show()
    
    for Y_axes in range(frame_hsv.shape[1]-1):
        for X_axes in range(frame_hsv.shape[0]-1):
            if frame_hsv[X_axes][Y_axes][1] > 30:
                frame_bin[X_axes][Y_axes] = 1
            else:
                frame_bin[X_axes][Y_axes] = 0

    frame_bin[frame_bin==1] = 255

    plt.imshow(frame_bin, cmap='gray')
    plt.show()
    
    
    ret, frame_thresh = cv2.threshold(frame_bin,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    x = 654157
    

    # for get_coi in range(len(colors_mean_cv)):
    #     if colors_mean_cv[get_coi][1] > 20:
    #         coi = colors_mean_cv[get_coi]
    #         break

    # threshold_upper_1 = np.array([coi[0]+2, 255, 255])
    # threshold_lower_1 = np.array([coi[0]-2, 0, 0])

    # threshold_mask_1 = cv2.inRange(frame_hsv, threshold_lower_1, threshold_upper_1)
    # # frame_threshold = cv2.bitwise_or(threshold_mask_1, threshold_mask_2))

    # utils.show_img(threshold_mask_1, 'Binarization')
    # # utils.save_img(threshold_mask_1, 'Binarization', date_stamp)

    # ------------------------------------------------------------------------------------------- Erode/Dilate
    frame_bin = cv2.cvtColor(frame_bin, cv2.COLOR_BGR2GRAY)
    ret, frame_thresh = cv2.threshold(frame_mean,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # frame_closing = cv2.morphologyEx(
    #     frame_mean, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))

    # frame_closing = cv2.erode(
    #     frame_closing, np.ones((5, 5), np.uint8), iterations=1)
    # frame_closing = cv2.dilate(frame_closing, np.ones(
    #     (13, 13), np.uint8), iterations=1)


    utils.show_img(frame_thresh, 'closing')
    # utils.save_img(frame_closing, 'Closing', date_stamp)




print(f'{(time.time() - start_time)} segundos gastos')
sys.stdout = orig_stdout
b.close()
