import cv2
import numpy as np
from gui_buttons import Buttons

buttons = Buttons()

buttons.add_button('Person', 20, 20)
buttons.add_button('Cell phone', 20, 100)
buttons.add_button('Remote', 20, 180)
net = cv2.dnn.readNet('models/yolov4-tiny.weights', 'models/yolov4-tiny.cfg')

model = cv2.dnn_DetectionModel(net)

model.setInputParams(size=(416, 416), scale=1 / 255)

classes = []

with open('models/classes.txt', 'r') as file_object:
    for class_name in file_object.readlines():
        classes.append(class_name.strip())

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def click_button(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        buttons.button_click(x, y)


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)
while True:
    ret, frame = cap.read()
    active_buttons = buttons.active_buttons_list()
    (class_ids, scores, bboxes) = model.detect(frame)

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        if classes[class_id] in active_buttons:
            cv2.putText(frame, str(classes[class_id]), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 50), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

    # create a button
    # cv2.rectangle(frame, (20, 20), (220, 70), (0, 0, 200), -1)

    # polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
    #
    # cv2.fillPoly(frame, polygon, (0, 0, 200))
    #
    # cv2.putText(frame, 'Person', (30, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    buttons.display_buttons(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if  key==27:
        break
cap.release()
cv2.destroyAllWindows()