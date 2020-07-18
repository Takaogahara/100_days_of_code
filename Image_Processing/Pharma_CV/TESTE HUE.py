__author__ = "Rafael Lopes Almeida"
__maintainer__ = "Rafael Lopes Almeida"
__email__ = "fael.rlopes@gmail.com"
__copyright__ = "None"
__license__ = "None"

__date__ = "22/05/2020"
__version__ = "0.2.0"
__revision__ = "4"
__status__ = "Prototype"
# -------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import cv2

import datetime
import platform
import sys
system_os = platform.system()

import utils

from math import sqrt
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# ------------------------------------------------------------------------------------------- Captura de imagem usando camera

def camera_capture(index=0, WIDTH=640, HEIGHT=480):
    cap = cv2.VideoCapture(index)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    # cap.set(10, BRIGHTNESS)
    # cap.set(11, CONTRAST)
    # cap.set(12, SATURATION)

    _, frame = cap.read()
    cap.release()

    return frame

# ------------------------------------------------------------------------------------------- Save/Show Image

def save_img(imagem, tipo ,date_stamp='null', formato='png', OS=system_os):
    if OS == 'Windows':
        cv2.imwrite('C:/Users/Rafael/Code/Projetos/Artigo/master/output/' +
                    date_stamp + '_' + str(tipo) + '.' + str(formato), imagem)
    elif OS == 'Linux':
        cv2.imwrite('C:/Users/Rafael/Code/Projetos/Artigo/master/output/' +
                    date_stamp + '_' + str(tipo) + '.' + str(formato), imagem)

# --------------------------------------------

def show_img(imagem, titulo):
    cv2.imshow(str(titulo), imagem)
    cv2.waitKey()
    cv2.destroyAllWindows()

# -------------------------------------------- Exportar terminal para TXT e inicializar csv

# def open_txt_csv():
    # import sys
    # orig_stdout = sys.stdout

    # if system_os == 'Windows':
    #     b = open('C:/Users/Rafael/Code/VS_Code/artigo_reloaded/master/log/log_terminal.txt', 'a')
    # elif system_os == 'Linux':
    #     b = open('C:/Users/Rafael/Code/VS_Code/artigo_reloaded/master/log/log_terminal.txt', 'a')
    # sys.stdout = b

    # sys.stdout = orig_stdout
    # b.close()
    
    # file_csv = open('C:/Users/Rafael/Code/VS_Code/Projeto/Logs/Result_Log.csv', 'a', newline='')
    # df.to_csv(file_csv, sep=";", header=False)
    # file_csv.close()

# ------------------------------------------------------------------------------------------- OPERACOES

def adjust_gamma(imagem, gamma=1.0):

    inv_Gamma = 1.0 / gamma
    gamma_table = np.array([((i / 255.0) ** inv_Gamma) * 255
         for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(imagem, gamma_table)

# --------------------------------------------

def apply_clache(imagem, clip_Limit=2.0, GridSize=8):
    frame_lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    frame_lab_planes = cv2.split(frame_lab)

    frame_clahe = cv2.createCLAHE(clipLimit=clip_Limit, tileGridSize=(GridSize, GridSize))

    frame_lab_planes[0] = frame_clahe.apply(frame_lab_planes[0])
    frame_lab = cv2.merge(frame_lab_planes)
    frame = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)

    return frame

# --------------------------------------------

def k_mean(imagem):
    # Lista de pixeis
    image_list = imagem.reshape((imagem.shape[0] * imagem.shape[1], 3))

    # data = np.array(image_list)
    # length = data.shape[0]
    # width = data.shape[1]
    # x, y = np.meshgrid(np.arange(length), np.arange(width))

    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.plot_surface(x, y, data)
    # plt.show()

    wcss = []
    for kmeans_iter in range(3, 10):
        kmeans_clf_wcss = KMeans(n_clusters = kmeans_iter, init = 'k-means++', n_init=3, max_iter=10)
        kmeans_clf_wcss.fit((image_list))
        wcss.append(kmeans_clf_wcss.inertia_)
    K = optimal_number_of_clusters(wcss)


    kmeans_clf = KMeans(n_clusters = K, init = 'k-means++', n_init=3, max_iter=10, )
    kmeans_clf.fit((image_list))
    image_kmeans = kmeans_clf.predict((image_list))
    
    image_kmeans = image_kmeans.reshape((imagem.shape[0], imagem.shape[1]))

    return image_kmeans

def optimal_number_of_clusters(wcss):
    x1, y1 = 1, wcss[0]
    x2, y2 = 19, wcss[len(wcss)-1]
    distances = []
    for i in range(len(wcss)):
        x0 = i + 1
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 1

# --------------------------------------------

def color_segmentation(imagem, space='rgb1', cor_1_inf=[0, 0, 0], cor_1_sup=[0, 0, 0], cor_2_inf=[0, 0, 0], cor_2_sup=[0, 0, 0]):
    if space == 'opencv':
        pass

    elif space == 'hsv':
        cor_1_inf[0] = int((cor_1_inf[0]/360) * 180)
        cor_1_inf[1] = int((cor_1_inf[1]/100) * 255)
        cor_1_inf[2] = int((cor_1_inf[2]/100) * 255)
        cor_1_sup[0] = int((cor_1_sup[0]/360) * 180)
        cor_1_sup[1] = int((cor_1_sup[1]/100) * 255)
        cor_1_sup[2] = int((cor_1_sup[2]/100) * 255)

        cor_2_inf[0] = int((cor_2_inf[0]/360) * 180)
        cor_2_inf[1] = int((cor_2_inf[1]/100) * 255)
        cor_2_inf[2] = int((cor_2_inf[2]/100) * 255)
        cor_2_sup[0] = int((cor_2_sup[0]/360) * 180)
        cor_2_sup[1] = int((cor_2_sup[1]/100) * 255)
        cor_2_sup[2] = int((cor_2_sup[2]/100) * 255)

    elif space == 'rgb1':
        r = cor_1_inf[0]
        g = cor_1_inf[1]
        b = cor_1_inf[2]

        r, g, b = r / 255.0, g / 255.0, b / 255.0
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        diff = cmax-cmin

        if cmax == cmin:
            h = 0
        elif cmax == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif cmax == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        elif cmax == b:
            h = (60 * ((r - g) / diff) + 240) % 360
        # if cmax == 0:
        #     s = 0
        # else:
        #     s = (diff / cmax) * 100
        # v = cmax * 100

        opencv_H = (h/360) * 180
        # opencv_S = (s/100) * 255
        # opencv_V = (v/100) * 255

        cor_1_inf[0] = int(opencv_H-3)
        cor_1_inf[1] = (0)
        cor_1_inf[2] = (0)
        cor_1_sup[0] = int(opencv_H+3)
        cor_1_sup[1] = (255)
        cor_1_sup[2] = (255)
    else:
        return print('Espaco de cor invalido')

    img_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    limite_inferior_1 = np.array([cor_1_inf[0], cor_1_inf[1], cor_1_inf[2]])
    limite_superior_1 = np.array([cor_1_sup[0], cor_1_sup[1], cor_1_sup[2]])
    limite_inferior_2 = np.array([cor_2_inf[0], cor_2_inf[1], cor_2_inf[2]])
    limite_superior_2 = np.array([cor_2_sup[0], cor_2_sup[1], cor_2_sup[2]])

    mascara_1 = cv2.inRange(img_hsv, limite_inferior_1, limite_superior_1)
    mascara_2 = cv2.inRange(img_hsv, limite_inferior_2, limite_superior_2)
    img_segmented = cv2.bitwise_or(mascara_1, mascara_2)

    return img_segmented

# --------------------------------------------

def find_contours_solid(imagem):

    area_contours_solid = []
    _, contours, _ = cv2.findContours(imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dummy = np.zeros((imagem.shape[0], imagem.shape[1], 3), dtype=np.uint8)
    for iter_find_contours in contours:
        x, y, w, h = cv2.boundingRect(iter_find_contours)
        cv2.rectangle(dummy, (x-0, y-0), (x+w+0, y+h+0), (0, 255, 0), -1)

    frame_contours_solid = cv2.inRange(dummy, np.array([0, 255, 0]), np.array([0, 255, 0]))
    len_contours_solid = len(contours)

    for iter_area in range(len_contours_solid):
        area_contours_solid.append(cv2.contourArea(contours[iter_area]))

    return frame_contours_solid, len_contours_solid, area_contours_solid

# --------------------------------------------

def find_contours(imagem):

    area_contours = []
    _, contours, _ = cv2.findContours(imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dummy = np.zeros((imagem.shape[0], imagem.shape[1], 3), dtype=np.uint8)
    for iter_find_contours in contours:
        x, y, w, h = cv2.boundingRect(iter_find_contours)
        cv2.rectangle(dummy, (x-0, y-0), (x+w+0, y+h+0), (0, 255, 0), 1)

    frame_contours = cv2.inRange(dummy, np.array([0, 255, 0]), np.array([0, 255, 0]))
    len_contours = len(contours)

    for iter_area in range(len_contours):
        area_contours.append(cv2.contourArea(contours[iter_area]))

    return frame_contours, len_contours, area_contours