# import the necessary packages
import argparse
import numpy as np
import cv2
import test
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str,
	default="haarcascade_frontalface_default.xml",
	help="path to haar cascade face detector")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])
vid = cv2.VideoCapture(0)
emotions = ["anger", "Happy", "Sad", "Surprise", "Neutral"]
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = cv2.resize(frame, (250, 250))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = detector.detectMultiScale(gray, scaleFactor=1.05,
		minNeighbors=5, minSize=(70, 70),
		flags=cv2.CASCADE_SCALE_IMAGE)
    # Display the resulting frame
    if faces != []:
        for face in faces:
            x,y,w,h = face
            image = gray[y:y+h, x:x+w]
            # draw the face bounding box on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image = cv2.resize(image, (48,48))
            image = image.reshape(48,48,1)
            emotion = emotions[test.predict_emotion(np.array([image]))[0]]
            cv2.putText(frame, emotion, (x-25, y+h+20), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
   
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
cv2.destroyAllWindows()