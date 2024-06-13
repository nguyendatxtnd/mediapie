import cv2
import mediapipe as mp

def hands():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    draw_spec_1 = mp_drawing.DrawingSpec(color = (0, 0, 255),thickness = 1,circle_radius = 1)
    draw_spec_2 = mp_drawing.DrawingSpec(color =(0, 0, 0),thickness = 1,circle_radius = 1)

    image = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False,
               max_num_hands=1,
               model_complexity=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5) as hands:
        while image.isOpened():
            ret,img = image.read()
            if not ret:
                print("read image not success")
            else:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                result = hands.process(img)
                img.flags.writeable = True
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img,hands
                                            connections=mp_hands.HAND_CONNECTIONS,
                                            landmark_drawing_spec: Optional[
                                                Union[DrawingSpec, Mapping[int, DrawingSpec]]
                                            ] = DrawingSpec(color=RED_COLOR),
                                            connection_drawing_spec: Union[
                                                DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]
                                            ] = DrawingSpec(),
                                            is_drawing_landmarks: bool = True,
                                        )
                
            cv2.imshow("video",img)
            k = cv2.waitKey(1)
            if k == ord("q") or k == ord ("Q"):
                break
        cv2.destroyAllWindows()

if __name__=="__main__":
    hands()

