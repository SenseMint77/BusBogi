import cv2 
import numpy as np
import threading
import time
import pytesseract
from PIL import Image


class Detection(threading.Thread):

    def __init__(self, bus_list):
        threading.Thread.__init__(self)
        self.dec_bus_list = bus_list

    def run(self):
        real_point = np.array([ [0,0] , [100,0] , [0, 150], [100,150]])
        image_point = np.array( [[219,437], [596,385],[170,183],[390,176]])
        H, status = cv2.findHomography(image_point, real_point, cv2.RANSAC)

        startTime = time.time()

        net = cv2.dnn.readNet('/Users/gino/PycharmProjects/Final_Project/BusBogi/api/yolo-data/yolo-bus-tiny.weights', '/Users/gino/PycharmProjects/Final_Project/BusBogi/api/yolo-data/yolov3-tiny_custom.cfg')

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        

        checkTime = time.time()
        cur_position = 0

        while True:
            cap = cv2.VideoCapture("http://192.168.103.58/html/cam_pic_new.php")  
            ret ,img  = cap.read()
            height, width, channels = img.shape

            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)

            font = cv2.FONT_HERSHEY_PLAIN

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    centerPoint = np.array([ x+w/2, y+h, 1])
                    ground_point = np.dot(H, centerPoint)
                    xd , yd = ground_point[0]/ground_point[2] , ground_point[1]/ground_point[2]

                    cur_position = int(yd)
                    
                    label = "bus : " + str(cur_position) + "m"

                    color = (180,20,20)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

                    cut_image = img[y:(y+h), x:(x+w) ]
                    resize_bus = cv2.resize(cut_image, dsize=(250, 420), interpolation=cv2.INTER_AREA)
                    height, width, channel = resize_bus.shape

                    b, g, r = cv2.split(resize_bus)
                    img = cv2.merge([r, g, b])

                    # Bilateral Filtering
                    dst4 = cv2.bilateralFilter(img, 3, 5, 50)

                    # from color to grey
                    gray_img = cv2.cvtColor(dst4, cv2.COLOR_BGR2GRAY)

                    # sharpening (필터)
                    kernel_sharpen = np.array(
                        [[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]]) / 8.0  # 정규화위해 8로나눔

                    output = cv2.filter2D(gray_img, -1, kernel_sharpen)

                    # Morphological gradient
                    kernal = np.ones((2, 2), np.uint8)
                    result0 = cv2.morphologyEx(output, cv2.MORPH_GRADIENT, kernal)

                    # Threshold
                    ret, thresh = cv2.threshold(result0, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    result = ''
                    # Find contours
                    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    for i in range(len(contours)):
                        x, y, w, h = cv2.boundingRect(contours[i])
                        aspect_ratio = float(w) / h
                        if (cv2.isContourConvex(contours[i]) == False) and hierachy[0][i][
                            2] != -1 and 1 < aspect_ratio < 2:
                            cnt = contours[i]

                            des = cv2.moments(cnt)
                            cx = int(des['m10'] / des['m00'])
                            cy = int(des['m01'] / des['m00'])

                            if cx < 180 and cy > 210 and 500 < des['m00']:

                                img = cv2.drawContours(gray_img, [cnt], -1, (0, 255, 0), 3)

                                x, y, w, h = cv2.boundingRect(cnt)
                                resize_bus = cv2.rectangle(resize_bus, (x, y), (x + w, y + h), (255, 0, 0), 3)
                                pts1 = np.float32([(x, y), (x, y + h), (x + w, y), (x + w, y + h)])

                                # 좌표의 이동점
                                pts2 = np.float32([[10, 10], [10, 80], [160, 10], [160, 80]])
                                M = cv2.getPerspectiveTransform(pts1, pts2)

                                dst = cv2.warpPerspective(resize_bus, M, (175, 90))

                                sharpening = np.array(
                                    [[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 9, 2, -1], [-1, 2, 2, 2, -1],
                                     [-1, -1, -1, -1, -1]]) / 9.0
                                dst_with_sharp = cv2.bilateralFilter(dst, 3, 5, 50)
                                res = cv2.filter2D(dst_with_sharp, -1, sharpening)

                                image = Image.fromarray(res)

                                config = ('-l eng --oem 1 --psm 8')

                                text = pytesseract.image_to_string(image, config=config)

                                for i in range(0, len(text)):
                                    if str.isdigit(text[i]):
                                        result += text[i]
                                if len(result) >= 3:
                                    print(result)
        return result


    def extract_number(self, cut_image):
        # resize
        resize_bus = cv2.resize(cut_image, dsize=(250, 420), interpolation=cv2.INTER_AREA)
        height, width, channel = resize_bus.shape

        b, g, r = cv2.split(resize_bus)
        img = cv2.merge([r, g, b])

        # Bilateral Filtering
        dst4 = cv2.bilateralFilter(img, 3, 5, 50)

        # from color to grey
        gray_img = cv2.cvtColor(dst4, cv2.COLOR_BGR2GRAY)

        # sharpening (필터)
        kernel_sharpen = np.array([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1],
                                   [-1, -1, -1, -1, -1]]) / 8.0  # 정규화위해 8로나눔

        output = cv2.filter2D(gray_img, -1, kernel_sharpen)

        # Morphological gradient
        kernal = np.ones((2, 2), np.uint8)
        result0 = cv2.morphologyEx(output, cv2.MORPH_GRADIENT, kernal)

        # Threshold
        ret, thresh = cv2.threshold(result0, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = ''
        # Find contours
        contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            aspect_ratio = float(w) / h
            if (cv2.isContourConvex(contours[i]) == False) and hierachy[0][i][2] != -1 and 1 < aspect_ratio < 2:
                cnt = contours[i]

                des = cv2.moments(cnt)
                cx = int(des['m10'] / des['m00'])
                cy = int(des['m01'] / des['m00'])

                if cx < 180 and cy > 210 and 500 < des['m00']:

                    img = cv2.drawContours(gray_img, [cnt], -1, (0, 255, 0), 3)

                    x, y, w, h = cv2.boundingRect(cnt)
                    resize_bus = cv2.rectangle(resize_bus, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    pts1 = np.float32([(x, y), (x, y + h), (x + w, y), (x + w, y + h)])

                    # 좌표의 이동점
                    pts2 = np.float32([[10, 10], [10, 80], [160, 10], [160, 80]])
                    M = cv2.getPerspectiveTransform(pts1, pts2)

                    dst = cv2.warpPerspective(resize_bus, M, (175, 90))

                    sharpening = np.array(
                        [[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 9, 2, -1], [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]]) / 9.0
                    dst_with_sharp = cv2.bilateralFilter(dst, 3, 5, 50)
                    res = cv2.filter2D(dst_with_sharp, -1, sharpening)

                    image = Image.fromarray(res)

                    config = ('-l eng --oem 1 --psm 8')

                    text = pytesseract.image_to_string(image, config=config)

                    for i in range(0, len(text)):
                        if str.isdigit(text[i]):
                            result += text[i]

                    print(result)

        return result


if __name__ == '__main__':

    bust  = []
    test = Detection(bust)
    test.daemon = False
    test.start()

    time.sleep(10)

    print("end!!")






