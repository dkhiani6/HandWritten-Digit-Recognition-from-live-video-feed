import cv2
import numpy as np
from sklearn.model_selection import PredefinedSplit
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D
import matplotlib.pyplot as plt

#------------------Training CNN------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

layer_1 = Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1))
model.add(layer_1)
layer_2 = Conv2D(64, kernel_size=3, activation='relu')
model.add(layer_2)
layer_3 = Flatten()
model.add(layer_3)
layer_4 = Dense(10, activation='softmax')
model.add(layer_4)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
print(model.summary())
#--------------------------------------Processing Video------------------------------------------
def process_frames(vid):
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    out = cv2.VideoWriter('saved_vid_out.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), 50, (frame_width,frame_height))
    while (True):
        ret, frame = vid.read()
        if ret==True:
            cv2.imshow("Original Video", frame)
            cv2.waitKey(1)
            grey = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY) # converting to gray scale
            cv2.imshow("Grey", grey)
            # cv2.waitKey(1)
            ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
            contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame.copy(), contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            # print(len(contours))
            for c in contours:
                if cv2.contourArea(c)>300 and cv2.contourArea(c)<9000:   #eliminating non-digit contours
                    x,y,w,h = cv2.boundingRect(c)
                
                # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
                    frame = cv2.rectangle(frame, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
                    cv2.imshow("Bounding box", frame)
                    # cv2.waitKey(1)
                # Cropping out the digit from the image corresponding to the current contours in the for loop
                    digit = thresh[y:y+h, x:x+w]
                    # cv2.imshow("Cropped",digit)
                    # cv2.waitKey(1)
                # Resizing that digit to (18, 18)
                    resized_digit = cv2.resize(digit, (18,18))
                    cv2.imshow("Resized", resized_digit)
                    cv2.waitKey(1)
                # Padding each digit with 5 pixels of black on each side so that corner pixels are not ignored while predicting
                # to finally produce image of size (28, 28)
                    padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
                
                # Adding the preprocessed digit to the list of preprocessed digits
                    # preprocessed_digits.append(padded_digit)
                    float_digit = np.array(padded_digit).astype(np.float32)
                    ans = model.predict(float_digit.reshape(1,28,28,1))
                    str_ans = str(np.argmax(ans))
                    #print(str_ans)
                    frame = cv2.putText(frame,str_ans, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2,cv2.LINE_AA)
                    #cv2.imshow("Predicted", frame)
                    out.write(frame)
                    cv2.imshow("Saved",frame)
        else:
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()
#-------------------------------------------Taking Input------------------------------------------------
print("1. Use WebCam to capture Live video\n2. Use pre-recorded video")
choice = int(input("Enter your choice: "))

if choice==1:
    vid = cv2.VideoCapture(0)
    process_frames(vid)
else:
    vid = cv2.VideoCapture('last4digits.mp4')
    process_frames(vid)
