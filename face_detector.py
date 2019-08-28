import cv2 as cv
import face_recognition
import numpy as np
import time

# Указываем какой камерой будем пользоваться
cap = cv.VideoCapture("/dev/video0")

if not cap.isOpened():
    print("Не удалось подключиться к камере") 
    exit()

# Подготавливаем переменную для хранения областей лиц
face_locations = []
# Подготавливаем переменную для хранения 
face_landmarks_list = []

# Параметры для контраста и яркости
alpha = 1.5 # Simple contrast control
beta = 50    # Simple brightness control

# # Initialize values
# print(' Basic Linear Transforms ')
# print('-------------------------')
# try:
#     alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
#     beta = int(input('* Enter the beta value [0-100]: '))
# except ValueError:
#     print('Error, not a number')


while True:
	
    # -----
    # Берем кадр с камеры
    # -----
    ret, frame = cap.read()

    if not ret:
        print("Не удается получить кадр")
        break

    # -----
    # Изменяем контраст и яркость
    # -----
    new_frame = np.zeros(frame.shape, frame.dtype)
    result = cv.addWeighted(frame, alpha, new_frame, 0, beta)

    # -----
    # Переводим формат картинки из BGR в RGB так как его использует face_recognition
    # -----
    rgb_frame = frame[:, :, ::-1]
            
    # Находим лица на кадре с помощью face_locations
    face_locations = face_recognition.face_locations(rgb_frame)
            
    # Наносим на кадр прямоугольники областей где обнаружены лица
    # for top, right, bottom, left in face_locations:
    #     # Рисуем прямоугольники вокруг лиц
    #     cv.rectangle(result, (left, top), (right, bottom), (0, 0, 255), 2)
    #     font = cv.FONT_HERSHEY_SIMPLEX
    #     cv.putText(result, 'Face', (left, top-5), font, 1, (0, 0, 255))
    
    # -----
    # Находим фичи лица на кадре
    # -----
    face_landmarks_list = face_recognition.face_landmarks(result)

    # Выводим информацию о фичах лица в консоль
    for face_landmarks in face_landmarks_list:

        # Выводим в консоль обнаруженные фичи лица
        # for facial_feature in face_landmarks.keys():
        #     print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

        # Рисуем полигоны вокруг обноруженных фич лица
        for facial_feature in face_landmarks.keys():
            pts = np.array(face_landmarks[facial_feature], np.int32)
            pts = pts.reshape((-1,1,2))
            cv.polylines(result, [pts], True, (0,255,255))

            # Делаем подпись крайнего полигона
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(result, facial_feature, ([pts][0][0][0][0],[pts][0][0][0][1]), font, 0.5, (0, 0, 255))
            
        
    # пробую рисовать полигон
    
    # pts=np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
    # pts = pts.reshape((-1,1,2))
    # cv.polylines(result, [pts], True, (0, 255, 255))
        
    # Выводим кадр
    cv.imshow('Video', result)
    
    # Добавляем возможность остановить выведение по нажатию кнопки 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Отключаемся от камеры и закрываем все окна
cap.release()
cv.destroyAllWindows()