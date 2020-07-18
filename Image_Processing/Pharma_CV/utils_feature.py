__author__ = "Rafael Lopes Almeida"
__maintainer__ = "Rafael Lopes Almeida"
__email__ = "fael.rlopes@gmail.com"
__copyright__ = "None"
__license__ = "None"

__date__ = "14/07/2020"
__version__ = "0.2.0"
__revision__ = "5"
__status__ = "Prototype"
# -------------------------------------------------------------------------------------------

from sklearn.cluster import KMeans
from math import sqrt
import numpy as np
import pandas as pd
import cv2

import datetime
import platform
import sys
system_os = platform.system()


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


def save_img(imagem, tipo, date_stamp='null', formato='png', OS=system_os):
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

    frame_clahe = cv2.createCLAHE(
        clipLimit=clip_Limit, tileGridSize=(GridSize, GridSize))

    frame_lab_planes[0] = frame_clahe.apply(frame_lab_planes[0])
    frame_lab = cv2.merge(frame_lab_planes)
    frame = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)

    return frame

# --------------------------------------------


def k_mean(imagem, K_iter=2, criteria_iter=5, criteria_eps=1.0):
    # imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, criteria_iter, criteria_eps)

    Z = imagem.reshape((-1, 3))
    Z = np.float32(Z)

    wcss = []
    k_range = 10
    for kmeans_iter in range(3, k_range):
        compactness, _, _ = cv2.kmeans(
            Z, kmeans_iter, None, criteria, K_iter, cv2.KMEANS_PP_CENTERS)
        wcss.append(compactness)
    K = optimal_number_of_clusters(wcss, k_range)

    _, label, center = cv2.kmeans(
        Z, K, None, criteria, K_iter, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    img_kmean = res.reshape((imagem.shape))

    color_list = get_color_value(label, center)

    return img_kmean, color_list


def get_color_value(Kmean_label, Kmean_center):
    unique_label = len(np.unique(Kmean_label))

    num_labels = np.arange(0, unique_label + 1)
    hist_label, _ = np.histogram(Kmean_label, bins=num_labels)
    hist_label = hist_label.astype("float")
    hist_label /= hist_label.sum()

    centroid_values = {}
    for centroid_creator in range(0, (len(num_labels) - 1)):
        centroid_values[centroid_creator] = Kmean_center[centroid_creator]

    color_list = {}
    for color_creator in range(0, (len(num_labels) - 1)):
        color_list[hist_label[color_creator]] = centroid_values[color_creator]

    color_list_sort = sorted(color_list.items(), reverse=True)

    return color_list_sort


def optimal_number_of_clusters(wcss, k_range):
    x1, y1 = 3, wcss[0]
    x2, y2 = k_range-1, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i + 3
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)

    return distances.index(max(distances)) + 3

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
    _, contours, _ = cv2.findContours(
        imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dummy = np.zeros((imagem.shape[0], imagem.shape[1], 3), dtype=np.uint8)
    for iter_find_contours in contours:
        x, y, w, h = cv2.boundingRect(iter_find_contours)
        cv2.rectangle(dummy, (x-0, y-0), (x+w+0, y+h+0), (0, 255, 0), -1)

    frame_contours_solid = cv2.inRange(
        dummy, np.array([0, 255, 0]), np.array([0, 255, 0]))
    len_contours_solid = len(contours)

    for iter_area in range(len_contours_solid):
        area_contours_solid.append(cv2.contourArea(contours[iter_area]))

    return frame_contours_solid, len_contours_solid, area_contours_solid

# --------------------------------------------


def find_contours(imagem):

    area_contours = []
    _, contours, _ = cv2.findContours(
        imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dummy = np.zeros((imagem.shape[0], imagem.shape[1], 3), dtype=np.uint8)
    for iter_find_contours in contours:
        x, y, w, h = cv2.boundingRect(iter_find_contours)
        cv2.rectangle(dummy, (x-0, y-0), (x+w+0, y+h+0), (0, 255, 0), 1)

    frame_contours = cv2.inRange(dummy, np.array(
        [0, 255, 0]), np.array([0, 255, 0]))
    len_contours = len(contours)

    for iter_area in range(len_contours):
        area_contours.append(cv2.contourArea(contours[iter_area]))

    return frame_contours, len_contours, area_contours
