import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train/train")
    os.makedirs("data/test/test")
    os.makedirs("data/train/train/A")
    os.makedirs("data/train/train/B")
    os.makedirs("data/train/train/C")
    os.makedirs("data/train/train/D")
    os.makedirs("data/rain/train/E")
    os.makedirs("data/train/train/F")
    os.makedirs("data/train/train/G")
    os.makedirs("data/train/train/H")
    os.makedirs("data/train/train/I")
    os.makedirs("data/train/train/J")
    os.makedirs("data/train/train/K")
    os.makedirs("data/train/train/L")
    os.makedirs("data/train/train/M")
    os.makedirs("data/train/train/N")
    os.makedirs("data/train/train/O")
    os.makedirs("data/train/train/P")
    os.makedirs("data/train/train/Q")
    os.makedirs("data/train/train/R")
    os.makedirs("data/train/train/S")
    os.makedirs("data/train/train/T")
    os.makedirs("data/train/train/U")
    os.makedirs("data/train/train/V")
    os.makedirs("data/train/train/W")
    os.makedirs("data/train/train/X")
    os.makedirs("data/train/train/Y")
    os.makedirs("data/train/train/Z")
    os.makedirs("data/train/train/del")
    os.makedirs("data/train/train/nothing")
    os.makedirs("data/train/train/space")
    os.makedirs("data/test/test/A_test.jpg")
    os.makedirs("data/test/test/B_test.jpg")
    os.makedirs("data/test/test/C_test.jpg")
    os.makedirs("data/test/test/D_test.jpg")
    os.makedirs("data/test/test/E_test.jpg")
    os.makedirs("data/test/test/F_test.jpg")
    os.makedirs("data/test/test/G_test.jpg")
    os.makedirs("data/test/test/H_test.jpg")
    os.makedirs("data/test/test/I_test.jpg")
    os.makedirs("data/test/test/J_test.jpg")
    os.makedirs("data/test/test/K_test.jpg")
    os.makedirs("data/test/test/L_test.jpg")
    os.makedirs("data/test/test/M_test.jpg")
    os.makedirs("data/test/test/N_test.jpg")
    os.makedirs("data/test/test/O_test.jpg")
    os.makedirs("data/test/test/P_test.jpg")
    os.makedirs("data/test/test/Q_test.jpg")
    os.makedirs("data/test/test/R_test.jpg")
    os.makedirs("data/test/test/S_test.jpg")
    os.makedirs("data/test/test/T_test.jpg")
    os.makedirs("data/test/test/U_test.jpg")
    os.makedirs("data/test/test/V_test.jpg")
    os.makedirs("data/test/test/W_test.jpg")
    os.makedirs("data/test/test/X_test.jpg")
    os.makedirs("data/test/test/Y_test.jpg")
    os.makedirs("data/test/test/Z_test.jpg")
    os.makedirs("data/test/test/nothing_test.jpg")
    os.makedirs("data/test/test/space_test.jpg")



# Train or test
mode = 'train'
directory = 'data/' + mode + '/' + mode + '/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Getting count of existing images
    count = {'A': len(os.listdir(directory + "A")),
             'B': len(os.listdir(directory + "B")),
             'C': len(os.listdir(directory + "C")),
             'D': len(os.listdir(directory + "D")),
             'E': len(os.listdir(directory + "E")),
             'F': len(os.listdir(directory + "F")),
             # 'G': len(os.listdir(directory + "G")),
             # 'H': len(os.listdir(directory + "H")),
             # 'I': len(os.listdir(directory + "I")),
             # 'J': len(os.listdir(directory + "J")),
             # 'K': len(os.listdir(directory + "K")),
             # 'L': len(os.listdir(directory + "L")),
             # 'M': len(os.listdir(directory + "M")),
             # 'N': len(os.listdir(directory + "N")),
             # 'O': len(os.listdir(directory + "O")),
             # 'P': len(os.listdir(directory + "P")),
             # 'Q': len(os.listdir(directory + "Q")),
             # 'R': len(os.listdir(directory + "R")),
             # 'S': len(os.listdir(directory + "S")),
             # 'T': len(os.listdir(directory + "T")),
             # 'U': len(os.listdir(directory + "U")),
             # 'V': len(os.listdir(directory + "V")),
             # 'W': len(os.listdir(directory + "W")),
             # 'X': len(os.listdir(directory + "X")),
             # 'Y': len(os.listdir(directory + "Y")),
             # 'Z': len(os.listdir(directory + "Z")),
             # 'del': len(os.listdir(directory + "del")),
             # 'nothing': len(os.listdir(directory + "nothing")),
             # 'space': len(os.listdir(directory + "space")),


             }

    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : " + mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(frame, "A : " + str(count['A']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(frame, "B : " + str(count['B']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(frame, "C : " + str(count['C']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(frame, "D : " + str(count['D']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(frame, "E : " + str(count['E']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(frame, "F : " + str(count['F']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "G : " + str(count['G']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "H : " + str(count['H']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "I : " + str(count['I']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "J : " + str(count['J']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "K : " + str(count['K']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "L : " + str(count['L']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "M : " + str(count['M']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "N : " + str(count['N']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "O : " + str(count['O']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "P : " + str(count['P']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "Q : " + str(count['Q']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "R : " + str(count['R']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "S : " + str(count['S']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "T : " + str(count['T']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "U : " + str(count['U']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "V : " + str(count['V']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "W : " + str(count['W']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "X : " + str(count['X']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "Y : " + str(count['Y']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "Z : " + str(count['Z']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "del : " + str(count['del']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "space : " + str(count['space']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    # cv2.putText(frame, "nothing : " + str(count['nothing']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)


    # Coordinates of the ROI
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (200, 200))


    cv2.imshow("Frame", frame)

    # _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((1, 1), np.uint8)
    # img = cv2.dilate(mask, kernel, iterations=1)
    # img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
    # roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    # _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

    cv2.imshow("ROI", roi)

#"Catching" keyboard interrupts

    key = cv2.waitKey(10)
    if key & 0xFF == 27:  # esc key
        break
    if key & 0xFF == ord('A'):
        cv2.imwrite(directory + 'A/A' + str(count['A']) + '.jpg', roi)
    elif key & 0xFF == ord('B'):
        cv2.imwrite(directory + 'B/B' + str(count['B']) + '.jpg', roi)
    elif key & 0xFF == ord('C'):
        cv2.imwrite(directory + 'C/' + str(count['C']) + '.jpg', roi)
    elif key & 0xFF == ord('D'):
        cv2.imwrite(directory + 'D/' + str(count['D']) + '.jpg', roi)
    elif key & 0xFF == ord('E'):
        cv2.imwrite(directory + 'E/' + str(count['E']) + '.jpg', roi)
    elif key & 0xFF == ord('F'):
        cv2.imwrite(directory + 'F/' + str(count['F']) + '.jpg', roi)









    # titles = ['Region Of Interest [ROI]', 'Frame']
    # images = [roi, frame]
    #
    # for i in range(2):
    #     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()

cap.release()
cv2.destroyAllWindows()
