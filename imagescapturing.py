import cv2
import os
import time
import uuid

IMAGES_PATH = 'Tensorflow/workspace/images/collectedimages'
labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
number_imgs = 15

for label in labels:
    # Create the directory if it doesn't exist
    label_path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error: Camera with index 3 could not be opened.")
        continue
    
    print('Collecting Images for {}'.format(label))
    time.sleep(5)
    
    for imgnum in range(number_imgs):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Frame could not be captured.")
            break
        
        imgname = os.path.join(label_path, '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
