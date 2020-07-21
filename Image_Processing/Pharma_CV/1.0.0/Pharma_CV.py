__author__ = "Rafael Lopes Almeida"
__maintainer__ = "Rafael Lopes Almeida"
__email__ = "fael.rlopes@gmail.com"

__date__ = "21/07/2020"
__version__ = "1.0.0"

# ------------------------------------------------------------------------------------------- Chamando libs
import utils
import cv2
import numpy as np

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
    try:
        frame_gamma = utils.adjust_gamma(frame, gamma=2.0)
    except:
        print('Erro: Gamma')

    utils.show_img(frame_gamma, 'Gamma')
    # utils.save_img(frame, 'Gamma', date_stamp)

    # ------------------------------------------------------------------------------------------- CLACHE
    try:
        frame_clache = utils.apply_clache(
            frame_gamma, clip_Limit=2.0, GridSize=8)
    except:
        print('Erro: Clache')

    # utils.show_img(frame_clache, 'Clache')
    # utils.save_img(frame, 'Clache', date_stamp)

    # ------------------------------------------------------------------------------------------- K-MEAN
    try:
        frame_mean, colors_mean = utils.k_mean(
            frame_clache, K_iter=5, criteria_iter=50, criteria_eps=50)
    except:
        print('Erro: K-Mean')

    utils.show_img(frame_mean, 'K Means')
    # utils.save_img(frame_mean, 'K-Mean', date_stamp)

    # ------------------------------------------------------------------------------------------- HSV Mask
    try:
        frame_segmented = utils.color_segmentation(
            frame_mean, 'bgr',
            cor_1_inf=[160, 0, 0])
    except:
        print('Erro: Segmentação de cores HSV')

    utils.show_img(frame_segmented, 'Mascara de cores')
    # utils.save_img(frame_segmented, 'Mascara', date_stamp)

    # ------------------------------------------------------------------------------------------- Erode/Dilate



print(f'{(time.time() - start_time)} segundos gastos')
sys.stdout = orig_stdout
b.close()
