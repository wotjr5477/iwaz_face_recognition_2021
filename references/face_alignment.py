Python 3.9.7 (tags/v3.9.7:1016ef3, Aug 30 2021, 20:19:38) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
import dlib # face_landmark >> 얼굴 좌표 68개
import cv2

RIGHT_EYE = list(range(36, 42)) # 오른쪽 눈 좌표 범위
LEFT_EYE = list(range(42, 48)) # 왼쪽 눈 좌표 범위
EYES = list(range(36, 48)) # 눈 좌표 범위

predictor_file = './model/shape_predictor_68_face_landmarks.dat'
image_file = './dataset/teedy/8.jpg'
MARGIN_RATIO = 1.5
OUTPUT_SIZE = (300, 300)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
image_origin = image.copy() # 카피된 이미지(변수)로 작업

(image_height, image_width) = image.shape[:2] # 이미지 높이와 폭 저장
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 노이즈 제거 > 흑백 사용

rects = detector(gray, 1) # 업스케일 레이어 수

def getFaceDimension(rect):
    return (rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top())

def getCropDimension(rect, center):
    width = (rect.right() - rect.left())
    half_width = width // 2
    (centerX, centerY) = center
    startX = centerX - half_width
    endX = centerX + half_width
    startY = rect.top()
    endY = rect.bottom() 
    return (startX, endX, startY, endY)    

for (i, rect) in enumerate(rects):
    (x, y, w, h) = getFaceDimension(rect) # OpenCV에 맞게 좌표 수정 (Dlib >> OpenCV)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # 얼굴에 사각형 그리기

    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()]) # 포인트 : 68개의 점(landmark) 가져옴
    show_parts = points[EYES] # 눈의 좌표만 보여줌

    right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype("int") # numpy로 좌표의 중간값(평균값)을 연산
    left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype("int")
    print(right_eye_center, left_eye_center)
 
    cv2.circle(image, (right_eye_center[0,0], right_eye_center[0,1]), 5, (0, 0, 255), -1) # 크기가 5인 빨간점을 눈 중앙(numpy로 연산)에 찍음
    cv2.circle(image, (left_eye_center[0,0], left_eye_center[0,1]), 5, (0, 0, 255), -1)
    
    cv2.circle(image, (left_eye_center[0,0], right_eye_center[0,1]), 5, (0, 255, 0), -1) # 삼각함수로 기울기를 구하기 위해서 x값과 y값을 각각 가져와서 나머지 좌표에서 90도를 이루는 임의의 점을 찍음
    
    cv2.line(image, (right_eye_center[0,0], right_eye_center[0,1]),
             (left_eye_center[0,0], left_eye_center[0,1]), (0, 255, 0), 2) # 두 눈 사이의 
    cv2.line(image, (right_eye_center[0,0], right_eye_center[0,1]),
         (left_eye_center[0,0], right_eye_center[0,1]), (0, 255, 0), 1) # 밑변
    cv2.line(image, (left_eye_center[0,0], right_eye_center[0,1]), 
         (left_eye_center[0,0], left_eye_center[0,1]), (0, 255, 0), 1) # 높이

    eye_delta_x = right_eye_center[0,0] - left_eye_center[0,0] # x변의 길이
    eye_delta_y = right_eye_center[0,1] - left_eye_center[0,1] # 높이
    degree = np.degrees(np.arctan2(eye_delta_y,eye_delta_x)) - 180 #아크-탄젠트(numpy)로 기울기 구하기 - 180 >> 각의 차이만큼 돌릴 예정이기 때문에 뻄

    eye_distance = np.sqrt((eye_delta_x ** 2) + (eye_delta_y ** 2)) # 두 눈 사이의 거리 구하기 피타고라스의 정리 활용
    aligned_eye_distance = left_eye_center[0,0] - right_eye_center[0,0] # 밑변
    scale = aligned_eye_distance / eye_distance # 눈 사이의 거리 >> 밑변의 거리가 되기 떄문에 scale을 구함

    eyes_center = ((left_eye_center[0,0] + right_eye_center[0,0]) // 2,
            (left_eye_center[0,1] + right_eye_center[0,1]) // 2) # 두 눈 사이의 좌표
    cv2.circle(image, eyes_center, 5, (255, 0, 0), -1) # 두 눈 사이에 파란 점
            
    metrix = cv2.getRotationMatrix2D(eyes_center, degree, scale) # 두 눈의 중간, degree, scale 를 metrix에 저장
    cv2.putText(image, "{:.5f}".format(degree), (right_eye_center[0,0], right_eye_center[0,1] + 20),
     	 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    warped = cv2.warpAffine(image_origin, metrix, (image_width, image_height), # 메트릭스값을 불러와서 이미지 회전
        flags=cv2.INTER_CUBIC)
    
    cv2.imshow("warpAffine", warped)
    (startX, endX, startY, endY) = getCropDimension(rect, eyes_center)
    croped = warped[startY:endY, startX:endX]
    output = cv2.resize(croped, OUTPUT_SIZE) # 300,300 사이즈로 리사이즈
    cv2.imshow("output", output)

    for (i, point) in enumerate(show_parts):
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)

cv2.imshow("Face Alignment", image)
cv2.waitKey(0)   
cv2.destroyAllWindows()
