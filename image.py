import cv2
import numpy as np
import math
import time
import os
from pathlib import Path
from cvzone.HandTrackingModule import HandDetector
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Failed to open camera")
            return

        detector = HandDetector(maxHands=2)
        
        OFFSET = 20
        IMG_SIZE = 300
        counter = 0

        # Use Path for cross-platform compatibility
        folder = Path("P:\RealTimeObjectDetection\Data\ILoveYou")
        folder.mkdir(parents=True, exist_ok=True)

        logging.info("Starting main loop")
        while True:
            success, img = cap.read()
            if not success:
                logging.warning("Failed to capture image")
                continue

            try:
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
                    imgCrop = img[max(0, y-OFFSET):min(img.shape[0], y+h+OFFSET),
                                  max(0, x-OFFSET):min(img.shape[1], x+w+OFFSET)]

                    if imgCrop.size == 0:
                        logging.warning("Cropped image is empty")
                        continue

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = IMG_SIZE / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                        wGap = math.ceil((IMG_SIZE - wCal) / 2)
                        imgWhite[:, wGap:wCal+wGap] = imgResize
                    else:
                        k = IMG_SIZE / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                        hGap = math.ceil((IMG_SIZE - hCal) / 2)
                        imgWhite[hGap:hCal+hGap, :] = imgResize

                    cv2.imshow('ImageCrop', imgCrop)
                    cv2.imshow('ImageWhite', imgWhite)

                cv2.imshow('Image', img)
                key = cv2.waitKey(1)
                
                if key == ord("s"):
                    counter += 1
                    filename = str(folder / f'Image_{time.time():.3f}.jpg')
                    cv2.imwrite(filename, imgWhite)
                    logging.info(f"Saved image {counter} to {filename}")
                elif key == ord("q"):
                    break
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        logging.info("Program ended")

if __name__ == "__main__":
    main()