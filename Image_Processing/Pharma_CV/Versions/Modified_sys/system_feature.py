__author__ = "Rafael Lopes Almeida"
__maintainer__ = "Rafael Lopes Almeida"
__email__ = "fael.rlopes@gmail.com"
__copyright__ = "None"
__license__ = "None"

__date__ = "21/07/2020"
__version__ = "0.2.3"
__revision__ = "5"
__status__ = "Prototype"

# ------------------------------------------------------------------------------------------- Chamando libs
import utils_feature
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

    # utils_feature.show_img(frame, 'Imagem')
    # utils_feature.save_img(frame, 'Cropped')

    # ------------------------------------------------------------------------------------------- Gamma
    try:
        frame_gamma = utils_feature.adjust_gamma(frame, gamma=2.0)
    except:
        print('Erro: Gamma')

    utils_feature.show_img(frame_gamma, 'Gamma')
    # utils_feature.save_img(frame, 'Gamma', date_stamp)

    # ------------------------------------------------------------------------------------------- CLACHE
    try:
        frame_clache = utils_feature.apply_clache(
            frame_gamma, clip_Limit=2.0, GridSize=8)
    except:
        print('Erro: Clache')

    # utils_feature.show_img(frame_clache, 'Clache')
    # utils_feature.save_img(frame, 'Clache', date_stamp)

    # ------------------------------------------------------------------------------------------- K-MEAN
    try:
        frame_mean, colors_mean = utils_feature.k_mean(
            frame_clache, K_iter=5, criteria_iter=50, criteria_eps=50)
    except:
        print('Erro: K-Mean')

    utils_feature.show_img(frame_mean, 'K Means')
    # utils_feature.save_img(frame_mean, 'K-Mean', date_stamp)

    # ------------------------------------------------------------------------------------------- HSV Mask
    try:
        frame_segmented = utils_feature.color_segmentation(
            frame_mean, 'bgr',
            cor_1_inf=[160, 0, 0])
    except:
        print('Erro: Segmentação de cores HSV')

    utils_feature.show_img(frame_segmented, 'Mascara de cores')
    # utils_feature.save_img(frame_segmented, 'Mascara', date_stamp)

    # ------------------------------------------------------------------------------------------- Erode/Dilate
    try:
        frame_closing = cv2.morphologyEx(
            frame_segmented, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))

        frame_closing = cv2.erode(
            frame_closing, np.ones((5, 5), np.uint8), iterations=1)
        frame_closing = cv2.dilate(frame_closing, np.ones(
            (13, 13), np.uint8), iterations=1)

    except:
        print('Erro: Erode/Dilate')

    # utils_feature.show_img(frame_closing, 'closing')
    # utils_feature.save_img(frame_closing, 'Closing', date_stamp)

    # ------------------------------------------------------------------------------------------- Filtro + Canny Edge
    try:
        # frame_blur_thresh = gaussian_blur_threshold(frame_closing, blur_coef=3, threshold_1=240, threshold_2=255)
        frame_canny = cv2.Canny(frame_closing, 100, 150)
    except:
        print('Erro: Canny Edge')

    # utils_feature.show_img(frame_canny, 'Canny Edge')
    # utils_feature.save_img(frame_canny, 'Canny', date_stamp)

    # ------------------------------------------------------------------------------------------- Contornos
    try:
        frame_contours_solid, _, _ = utils_feature.find_contours_solid(
            frame_canny)
        frame_contours, len_contours, area_contours = utils_feature.find_contours(
            frame_contours_solid)
    except:
        print('Erro: Contornos')

    # utils_feature.show_img(frame_contours_solid, 'Find Contours - Solid')
    # utils_feature.show_img(frame_contours, 'Find Contours', date_stamp)
    utils_feature.save_img(frame_contours, 'Contours')

    print(
        f'img: {current_iter}     Result: {len_contours} / {N_BLISTER * N_PILLS}')


print(f'{(time.time() - start_time)} segundos gastos')
sys.stdout = orig_stdout
b.close()
