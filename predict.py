import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# classes dictionary
classes = {'A' : 0,
           'B' : 1,
           'C' : 2,
           'D' : 3,
           'E' : 4,
           'F' : 5,
           'G' : 6,
           'H' : 7,
           'I' : 8,
           'J' : 9,
           'K' : 10,
           'L' : 11,
           'M' : 12,
           'N' : 13,
           'O' : 14,
           'P' : 15,
           'Q' : 16,
           'R' : 17,
           'S' : 18,
           'T' : 19,
           'U' : 20,
           'V' : 21,
           'W' : 22,
           'X' : 23,
           'Y' : 24,
           'Z' : 25}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]

    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (200, 200))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # _, test_image = roi
    # cv2.imshow("test", test_image)
    # 1 batch
    result = loaded_model.predict(roi.reshape(1, 200, 200, 3))
    prediction = {'A': result[0][0],
                  'B': result[0][1],
                  'C': result[0][2],
                  'D': result[0][3],
                  'E': result[0][4],
                  'F': result[0][5]}
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.imshow("Frame", frame)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break

cap.release()
cv2.destroyAllWindows()