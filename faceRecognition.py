import cv2
import numpy as np
import os
import time
import RPi.GPIO as GPIO

# 一些参数
servoPIN_y = 18     # 绕y轴转的舵机的控制引脚（也就是抬头低头的那个方向）
servoPIN_z = 23     # 绕z轴转的舵机的控制引脚（也就是摇头的那个方向）
FRAME_WIDTH = 640   # 相机画面的大小，越大运行速度越慢
FRAME_HEIGHT = 480
# OpenCV里面图片的坐标是这样的：左上角(0,0)右下角(MAX,MAX),所以相机画面中心的坐标就是(画面宽/2，画面高/2)
SCREEN_CENTER_X = FRAME_WIDTH / 2
SCREEN_CENTER_Y = FRAME_HEIGHT / 2


def setServoAngle(servoPIN, angle):
    '''
    函数功能：指定舵机转到某一角度
    servoPIN 控制舵机的引脚编号
    angle 舵机的位置，范围是：0到180，单位是度
    转向舵机的原理：
    云台上使用的是2个180度舵机，三个引脚分别是 棕---->GND 橙---->+5V/3.3V 黄---->方波信号
    显然控制舵机转向角度的只可能是方波信号的某个参数。舵机使用的方波信号周期是固定的20ms，所以控制转向角度的只有方波的占空比了，如果我们输入一个占空比2.5%的信号，舵机就转到0°，输入占空比12.5%的信号，舵机就转到180°，占空比和角度成线性。
    '''
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(servoPIN, GPIO.OUT)
    pwm = GPIO.PWM(servoPIN, 50)    # 方波频率固定50Hz
    pwm.start(7.5)
    dutyCycle = angle / 18. + 2.5   # 从角度算占空比
    pwm.ChangeDutyCycle(dutyCycle)  # 改方波占空比的函数ChangeDutyCycle
    # 一定要留时间给舵机转动，不然舵机不会动的。0.2s有可能少了，不过转动角度一般不会太大，所以应该也够用，留的时间太大的话画面容易不连贯
    time.sleep(0.2)
    pwm.stop()
    GPIO.cleanup()


# 初始化角度都是90°
setServoAngle(servoPIN_y, 90)
setServoAngle(servoPIN_z, 90)

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('/home/sychen/FaceRecognition/trainer.yml')
cascadePath = "/home/sychen/FaceRecognition/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['Bill_Duke', 'Charice', 'Elon_Musk', 'Jack_Ma', 'Zoe_Levin']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, FRAME_WIDTH)  # set Width
cam.set(4, FRAME_HEIGHT)  # set Height
angle_z = 90
angle_y = 90

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img = cam.read()
    img = cv2.flip(img, 1)   # cv2.flip函数用来镜像翻转画面，第二个参数表示怎么翻转，1表示仅做水平镜像翻转

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    '''
    faceCascade.detectMultiScale返回的结果是检测到的每个目标的（目标左上角x坐标,目标左上角y坐标，目标宽度，目标高度），每个目标的这四个数据保存在一个元组tuple里，多个元组构成列表list。例如，当前帧检测到2个目标，那么faces就是这样的东西：
    [(1, 2, 3, 4), (11, 22, 33, 44)]
    没检测到目标的话返回一个空元组。
    '''
    # 这个for循环是给检测到的每个目标分别画框框
    for(x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if (len(faces) != 0):
            # 如果检测到1个以上的目标怎么办？不管了我就取第一个，让聚光灯打在第一个目标上，lay了lay了。
            x, y, w, h = faces[0]
            face_center_x = x + w / 2   # 计算目标的中心，显然是目标左上角坐标 + 目标的宽(或高)/2
            face_center_y = y + h / 2
            # 如果目标中心离屏幕中心太远（这里“太远”的阈值是整个画面宽（或高）的10分之1，如果你想让舵机反应更快一点点，可以把10改大）
            # 到底转动多少度，才能让画面中心和目标中心重合呢？我不知道，好像这个数值和摄像机-人脸的距离有关系吧。这里舵机每次转动的角度是两个中心差值的0.03倍，这个数值是随便设置的，摄像头和人大概在0.5米的时候这个值感觉还行。根据感觉调吧。
            if (face_center_x - SCREEN_CENTER_X > FRAME_WIDTH / 10):
                angle_z += (face_center_x - SCREEN_CENTER_X) * 0.02
                if angle_z > 180:
                    angle_z = 180
                setServoAngle(servoPIN_z, angle_z)
            elif (SCREEN_CENTER_X - face_center_x > FRAME_WIDTH / 10):
                angle_z -= (SCREEN_CENTER_X - face_center_x)*0.02
                if angle_z < 0:
                    angle_z = 0
                setServoAngle(servoPIN_z, angle_z)
            if (face_center_y - SCREEN_CENTER_Y > FRAME_HEIGHT / 10):
                angle_y -= (face_center_y - SCREEN_CENTER_Y)*0.02
                if angle_y > 180:
                    angle_y = 180
                setServoAngle(servoPIN_y, angle_y)
            elif (SCREEN_CENTER_Y - face_center_y > FRAME_HEIGHT / 10):
                angle_y += (SCREEN_CENTER_Y - face_center_y)*0.02
                if angle_y < 0:
                    angle_y = 0
                setServoAngle(servoPIN_y, angle_y)

        id, error = recognizer.predict(gray[y:y+h, x:x+w])

        # 这里或许是你论文里面可以写一点东西的地方
        # predict()函数有两个返回值id和error，其中error表示函数预测当前帧的人脸属于id的误差是多大，但是官方文档并没有明确告诉我们error的取值范围，所以我们要自己想办法把这个误差转换成信心confidence。
        # 这里用的是很简单的处理方式：error为0时confidence=100-100/350*0=100，error为350时confidence=100-100/350*350=0，也就是说confidence和error在一定范围内成线性的反相关，在这里这个范围是（0,350）。你可以调大350这个数，调大他说明可以容忍更大的error，也就是说更容易把当前帧的人脸匹配上某个id，但同时也带来了误判的风险。
        # 网上关于error和百分比的转换的讨论也很多，比如
        # http://opencv-users.1802565.n2.nabble.com/Confidence-Level-based-on-Euclidean-Distance-td5876309.html
        # https://blog.csdn.net/iriszx999/article/details/79879578
        confidence = 100. - 100./350. * error
        if (confidence > 60):
            id = names[id]
            confidence = "  {0}%".format(round(confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(confidence))

        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5),
                    font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
