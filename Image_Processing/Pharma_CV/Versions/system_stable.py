__author__ = "Rafael Lopes Almeida"
__maintainer__ = "Rafael Lopes Almeida"
__email__ = "fael.rlopes@gmail.com"
__copyright__ = "None"
__license__ = "None"

__date__ = "22/05/2020"
__version__ = "0.2.0"
__revision__ = "4"
__status__ = "Prototype"

# ------------------------------------------------------------------------------------------- Chamando libs
import time
start_time = time.time()
import datetime
import cv2
import numpy as np
import pandas as pd
import platform
import sys
system_os = platform.system()

# ------------------------------------------------------------------------------------------- List/Dict/Counter
# ------------------------------------------------------------------------------------------- Parametros
N_BLISTER = 3
N_PILLS = 8

AREA_THRESHOLD = 0
MEAN_AREA = 0

# ------------------------------------------------------------------------------------------- Funcoes
# -------------------------------------------- Captura de imagem usando camera

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

# -------------------------------------------- Salvar imagens

def save_img(imagem, tipo, formato='png', OS=system_os):
    if OS == 'Windows':
        cv2.imwrite('C:/Users/Rafael/Code/VS_Code/artigo_reloaded/master/output/' +
                    date_stamp + '_' + str(tipo) + '.' + str(formato), imagem)
    elif OS == 'Linux':
        cv2.imwrite('C:/Users/Rafael/Code/VS_Code/artigo_reloaded/master/output/' +
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

# -------------------------------------------- OPERACOES

def apply_clache(imagem, clip_Limit=2.0, GridSize=8):
    frame_lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    frame_lab_planes = cv2.split(frame_lab)

    frame_clahe = cv2.createCLAHE(clipLimit=clip_Limit, tileGridSize=(GridSize, GridSize))

    frame_lab_planes[0] = frame_clahe.apply(frame_lab_planes[0])
    frame_lab = cv2.merge(frame_lab_planes)
    frame = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)

    return frame

# --------------------------------------------

def adjust_gamma(imagem, gamma=1.0):

    inv_Gamma = 1.0 / gamma
    gamma_table = np.array([((i / 255.0) ** inv_Gamma) * 255
         for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(imagem, gamma_table)

# --------------------------------------------

def k_mean(imagem, K=5, criteria_iter=5, criteria_eps=1.0, K_iter=2):
    Z = frame.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER, criteria_iter, criteria_eps)
    _, label, center = cv2.kmeans(Z, K, None, criteria, K_iter, cv2.KMEANS_PP_CENTERS)  # cv2.KMEANS_RANDOM_CENTERS

    center = np.uint8(center)
    res = center[label.flatten()]
    img_kmean = res.reshape((frame.shape))

    return img_kmean

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
        return print('Epaco de cor invalido')

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

# ------------------------------------------------------------------------------------------- MAIN

orig_stdout = sys.stdout

if system_os == 'Windows':
    b = open('C:/Users/Rafael/Code/VS_Code/artigo_reloaded/master/log/log_terminal.txt', 'a')
elif system_os == 'Linux':
    b = open('C:/Users/Rafael/Code/VS_Code/artigo_reloaded/master/log/log_terminal.txt', 'a')
sys.stdout = b


for current_iter in range(1, 200):
    # ------------------------------------------------------------------------------------------- Imagem
    # -------------------------------------------- Abrir imagem
    try:
        if system_os == 'Windows':
            frame = cv2.imread(f'C:/Users/Rafael/Code/VS_Code/artigo_reloaded/master/samples/neralgyn/{current_iter}.png', 1)
        elif system_os == 'Linux':
            frame = cv2.imread(f'C:/Users/Rafael/Code/VS_Code/artigo_reloaded/master/samples/neralgyn/{current_iter}.png', 1)
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

    # show_img(frame, 'Imagem')
    # save_img(frame, 'Cropped')

    # ------------------------------------------------------------------------------------------- Gamma
    try:
        frame_gamma = adjust_gamma(frame, gamma=2.0)
    except:
        print('Erro: Gamma')

    # show_img(frame_gamma, 'Gamma')
    # save_img(frame, 'Gamma')

    # ------------------------------------------------------------------------------------------- CLACHE
    try:
        frame_clache = apply_clache(frame_gamma, clip_Limit=2.0, GridSize=8)
    except:
        print('Erro: Clache')

    # show_img(frame_clache, 'Clache')
    # save_img(frame, 'Clache')

    # ------------------------------------------------------------------------------------------- K-MEAN
    # TODO: Aplicar "Elbow Method" para determinar numero de K
    try:
        frame_mean = k_mean(frame_clache, K=5, criteria_iter=10, K_iter=5)
    except:
        print('Erro: K-Mean')

    # show_img(frame_mean, 'K Means')
    # save_img(frame_mean, 'K-Mean')

    # ------------------------------------------------------------------------------------------- HSV Mask
    # TODO: Fazer uma maneira de pegar cor ideal do K-Means
    try:
        frame_segmented = color_segmentation(frame_mean, 'rgb1', [50, 34, 37])
    except:
        print('Erro: Segmentação de cores HSV')

    # show_img(frame_segmented, 'Mascara de cores')
    # save_img(frame_segmented, 'Mascara')

    # ------------------------------------------------------------------------------------------- Erode/Dilate
    try:
        frame_closing = cv2.morphologyEx(
            frame_segmented, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))

        frame_closing = cv2.erode(frame_closing, np.ones((5, 5), np.uint8), iterations=1)
        frame_closing = cv2.dilate(frame_closing, np.ones((13, 13), np.uint8), iterations=1)

    except:
        print('Erro: Erode/Dilate')

    # show_img(frame_closing, 'closing')
    # save_img(frame_closing, 'Closing')

    # ------------------------------------------------------------------------------------------- Filtro + Canny Edge
    try:
        # frame_blur_thresh = gaussian_blur_threshold(frame_closing, blur_coef=3, threshold_1=240, threshold_2=255)
        frame_canny = cv2.Canny(frame_closing, 100, 150)
    except:
        print('Erro: Canny Edge')

    # show_img(frame_canny, 'Canny Edge')
    # save_img(frame_canny, 'Canny')

    # ------------------------------------------------------------------------------------------- Contornos
    try:
        frame_contours_solid, _, _ = find_contours_solid(frame_canny)
        frame_contours, len_contours, area_contours = find_contours(frame_contours_solid)
    except:
        print('Erro: Contornos')

    # show_img(frame_contours_solid, 'Find Contours - Solid')
    # show_img(frame_contours, 'Find Contours')
    save_img(frame_contours, 'Contours')

    print(f'img: {current_iter}     Result: {len_contours} / {N_BLISTER * N_PILLS}')


print(f'{(time.time() - start_time)} segundos gastos')
sys.stdout = orig_stdout
b.close()