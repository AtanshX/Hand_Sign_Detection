import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size= 100
delay_ms = 200  # Adjust delay in milliseconds

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        img=cv2.flip(frame,1)
        cv2.putText(img, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', img)
        if cv2.waitKey(25) == ord('q'):
            break


    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        img = cv2.flip(frame, 1)
        cv2.imshow('frame', img)
        cv2.waitKey(delay_ms)
  # Introduce delay here
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), img)

        counter += 1

cap.release()
cv2.destroyAllWindows()