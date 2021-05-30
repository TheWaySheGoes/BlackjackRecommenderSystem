import cv2
import numpy as np
from PIL import ImageGrab

class Screen():

    #vid_cap = cv2.VideoWriter_fourcc(*'MPEG')
    #out = cv2.VideoWriter('output.avi', vid_cap, 8.0, (200, 200)) # If you want to record screen.
    cap = cv2.VideoCapture(0)

    def rescale_frame(frame, percent=75):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)



    while True:
        img = ImageGrab.grab()
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
      #  cv2.imshow("Screen", frame)
        #out.write(frame)
        
        #rect, frame = cap.read()
        frame75 = rescale_frame(frame, percent=75)
        cv2.imshow('Screen', frame75)


        if cv2.waitKey(1) == 27: # press escape to break
            break

    #out.release() save the video file.
    cv2.destroyAllWindows()