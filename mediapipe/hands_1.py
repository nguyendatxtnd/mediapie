import cv2
import mediapipe as mp

def hands_1():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    draw_spec = mp_drawing.DrawingSpec(color =(0, 128, 0), thickness = 3, circle_radius = 3)
    draw_spec_1 = mp_drawing.DrawingSpec(color =(0, 0, 255), thickness = 3, circle_radius = 3)
    a = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode = False,
                        max_num_hands = 2,
                         model_complexity=1,
                         min_detection_confidence=0.5,
                         min_tracking_confidence=0.5) as hands:
        
        while a.isOpened():
            ret,image = a.read()
            if not ret:
                print("read failed")
                break
            else:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image,hand,
                                            connections = mp_hands.HAND_CONNECTIONS,
                                            landmark_drawing_spec = draw_spec_1,
                                            connection_drawing_spec = draw_spec)   
            cv2.imshow(" ",image)
            k = cv2.waitKey(1)
            if k == ord("q") or k == ord("Q"):
                break
    a.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    hands_1()
                          
