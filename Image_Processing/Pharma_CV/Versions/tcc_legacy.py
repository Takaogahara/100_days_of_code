import time
import cv2
import numpy as np
import pandas as pd
from statistics import mean
import datetime
import sys
# from gpiozero import Button, LED

orig_stdout = sys.stdout
# b = open('VS_Code/Output/Terminal_Log.txt', 'a')
b = open('/home/pi/Python/Output/Terminal_Log.txt', 'a')
sys.stdout = b

# file_csv = open('VS_Code/Output/Result_Log.csv', 'a', newline='')
file_csv = open('/home/pi/Python/Output/Result_Log.csv', 'a')

nbr = 0
sum_area = 0
prod = 0
dic = {}
result = [0, 0, 0]
contours_2 = []
area = []

# sensor = Button(2)
# led_on = LED(17)
# led_run = LED(27)
# led_wait = LED(22)
# led_on.on()

# ------------------------------------------------------------------------
# CYCLES = 10
OFFSET = 0
AREA_THRESHOLD = 500
MEAN_AREA = 1487
N_PILLS = 8
N_BLISTER = 3

criteria_mode = (cv2.TERM_CRITERIA_EPS)
# (cv2.TERM_CRITERIA_EPS) | (cv2.TERM_CRITERIA_MAX_ITER) | (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER)
ITER = 3
EPS = 3.0
KITER = 2
#
COR_1 = [165, 0,0] #Amarelo 
COR_2 = [175,255,255]
#
COR_3 = [0, 0, 0] # Vermelho
COR_4 = [0, 0, 0]
#
K = 5

E = 3 
ET = 3

D = 3
DT = 2
#
WIDTH = 640
HEIGHT = 480
BRIGHTNESS = 50
CONTRAST = 0
SATURATION = 0
# ------------------------------------------------------------------------
# led_run.on()
# ------------------------------------------------------------------------ Abrir imagem
start_time = time.time()
# frame = cv2.imread('VS_Code/Fotos/live_feed.jpg',1)
# frame = cv2.imread('/home/pi/Python/Fotos/live_feed.jpg',1)
# cv2.imshow("Imagem original", frame)
# cv2.waitKey()
# cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
# cap.set(3, WIDTH)
# cap.set(4, HEIGHT)
# cap.set(10, BRIGHTNESS)
# cap.set(11, CONTRAST)
# cap.set(12, SATURATION)

_, frame = cap.read()
cap.release()

frame = frame [180:400, 120:500]
# cv2.imshow("Imagem cortada", frame)
# cv2.waitKey()
# cv2.destroyAllWindows()

file_name = str(datetime.datetime.now())
cv2.imwrite('/home/pi/Python/Fotos/Teste_3/Orig/' + file_name + '_original.png', frame)
print(file_name)

# ------------------------------------------------------------------------ Unsharp mask
# cv2.imwrite("frame_original.jpg", frame)
# gaussian_blur = cv2.GaussianBlur(frame, (5,5), 10.0)
# frame = cv2.addWeighted(frame, 1.5, gaussian_blur, -0.5, 0, frame)
# # cv2.imwrite("frame_unsharp.jpg", frame_unsharp)

# ------------------------------------------------------------------------ K-MEAN
Z = frame.reshape((-1,3))
Z = np.float32(Z)

criteria = (criteria_mode, ITER, EPS)
ret, label, center = cv2.kmeans(Z, K, None, criteria, KITER, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
frame = res.reshape((frame.shape))
# cv2.imshow('K Means',frame)
# cv2.waitKey()
# cv2.destroyAllWindows()

# ------------------------------------------------------------------------ HSV
img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

limite_inferior_1 = np.array([COR_1[0],COR_1[1],COR_1[2]])
limite_superior_1 = np.array([COR_2[0],COR_2[1],COR_2[2]])

limite_inferior_2 = np.array([COR_3[0],COR_3[1],COR_3[2]])
limite_superior_2 = np.array([COR_4[0],COR_4[1],COR_4[2]])

mascara_1 = cv2.inRange(img_hsv, limite_inferior_1, limite_superior_1)
mascara_2 = cv2.inRange(img_hsv, limite_inferior_2, limite_superior_2)

img = cv2.bitwise_or(mascara_1,mascara_2)
# cv2.imshow('Mascara',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

cv2.imwrite('/home/pi/Python/Fotos/Teste_3/K/' + file_name + '_K.png', frame)

# ------------------------------------------------------------------------ Erode
img = cv2.erode(img, np.ones((E,E), np.uint8), iterations=ET)
img = cv2.dilate(img, np.ones((D,D), np.uint8), iterations=DT) 
# cv2.imshow('Erode/Dilate', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# ------------------------------------------------------------------------ Filtro + Canny Edge
img_blur = cv2.blur(img,(3,3)) # Gaussian Blur
_, img_blur = cv2.threshold(img,240,255,cv2.THRESH_BINARY)

img_canny = cv2.Canny(img_blur,100,120) # Canny Edge
# cv2.imshow("Canny Edge",img_canny)
# cv2.waitKey()
# cv2.destroyAllWindows()

# ------------------------------------------------------------------------ Contornos
_, contours_1, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours_1:
    rect = cv2.boundingRect(c)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(frame, (x-0, y-0), (x+w+0, y+h+0), (230, 48, 181), 2)
    # cv2.drawContours(frame, c, -1, (230, 48, 181), 3)

limite_inferior_sqr = np.array([229,47,180])
limite_superior_sqr = np.array([231,49,182])
mascara_sqr = cv2.inRange(frame, limite_inferior_sqr, limite_superior_sqr)
# cv2.imshow('Contorno', mascara_sqr)
# cv2.waitKey()
# cv2.destroyAllWindows()

_, contours_x, _ = cv2.findContours(mascara_sqr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for f in range (len(contours_x)):
    shape = np.shape(contours_x[f])
    if (shape[0] >= 8) & (cv2.contourArea(contours_x[f]) > 700):
        nbr = nbr +1
        contours_2.append(contours_x[f]) 

for c in contours_2: 
    rect = cv2.boundingRect(c)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(frame, (x-0, y-0), (x+w+0, y+h+0), (0, 255, 0), 2)
    # cv2.drawContours(frame, c, -1, (230, 48, 181), 3)

print("Encontrados %d/%d objetos." % (len(contours_2), N_PILLS*N_BLISTER))       
# cv2.imshow('Contorno end', frame)
# cv2.waitKey()
# cv2.destroyAllWindows()

cv2.imwrite('/home/pi/Python/Fotos/Teste_3/end/' + file_name + '_final.jpg', frame)
 
result[0] = N_PILLS*N_BLISTER - len(contours_2)

# ------------------------------------------------------------------------ Area
for v in range (nbr):
    area.append(cv2.contourArea(contours_2[v]))
    print("\tArea do contorno %d: %d" % (v+1, area[v]))

media_area = mean(area)
print("\tMEDIA:  %d" % media_area)

for m in range (nbr):
    if (abs(area[m] - MEAN_AREA) > AREA_THRESHOLD):
        result[1] = result[1] + 1

# ------------------------------------------------------------------------ Database
dic.update({(prod - OFFSET) : result})
print(result)

# ------------------------------------------------------------------------ Arquivo CSV
df = pd.DataFrame.from_dict(dic)
df = df.T
df.to_csv(file_csv, sep=";", header=False)

# led_run.off()
# led_on.off()
print("---------------------------------------- %.8s sec <<<<<<<" % (time.time() - start_time))

sys.stdout = orig_stdout
b.close()

file_csv.close()

cv2.imshow('Contorno end', frame)
cv2.waitKey()
cv2.destroyAllWindows()