

import tensorflow as tf
import cv2
import numpy as np
import time
import winsound


from dataPreparation import prepare, GPSdata
import ThingsSpeak
import arduinoData

model = tf.keras.models.load_model("miscellaneous\drowiness_new6.h5")

vid = cv2.VideoCapture(0)
def drowsiness_output(prediction):
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    Lat, Long = GPSdata()
    if prediction == 0:
        winsound.Beep(frequency, duration)
        ThingsSpeak.post_data(0, Lat, Long)
        print("Driver is yawning.")
    elif prediction == 1:
        print("Driver is not yawning.")
        ThingsSpeak.post_data(1, Lat, Long)
    elif prediction == 2:
        winsound.Beep(frequency, duration)
        ThingsSpeak.post_data(2, Lat, Long)
        print("Drivers eyes are closed.")
    elif prediction == 3: 
        print("Drivers eyes are open.")
        ThingsSpeak.post_data(3, Lat, Long)
    else:
        print("Error.")

while True:
    ret, frame = vid.read()
    prediction = model.predict([prepare(frame)])
    drowsiness_output(np.argmax(prediction))
    time.sleep(5)