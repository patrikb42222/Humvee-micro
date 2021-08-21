import numpy as np
from PIL import ImageGrab
import cv2
import tensorflow as tf

micro_model = tf.keras.Sequential()
micro_model.add(tf.keras.Input(shape = (400,250,3)))
micro_model.add(tf.keras.Dense(32))
micro_model.add(tf.keras.Dense(32))
micro_model.add(tf.keras.Dense(32))
model.compile(optimizer='adam', loss='mse')
    

while (True):
    screenshot_pil = ImageGrab.grab(bbox=(0,25,800,625))
    cv2.imshow('window', np.array(screenshot_pil))
    if (cv2.waitKey(25) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        break

    