import cv2
import mediapipe as mp

def face_detection_1():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    a = cv2.VideoCapture(0)
    while True:
        ret,img = a.read()
        if not ret:
            print("read image not successed")
            break
        else:
            results=mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5).process(img)
        if results.detections:
            for detection in results.detections:
                confidence_score = detection.score[0]
                if confidence_score >= 0.5:
                    mp_drawing.draw_detection(img,detection)
        cv2.imshow("video",img)
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
    a.release()
    cv2.destroyAllWindows()
def main():
    face_detection_1()
if __name__ == '__main__':
    main()
    