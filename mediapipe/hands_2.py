import cv2
import mediapipe as mp

def detection_hand_gestures():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    draw_spec_1 = mp_drawing.DrawingSpec(color = (0,128,0), thickness = 8)
    draw_spec_2 = mp_drawing.DrawingSpec(color = (0,0,255), circle_radius=4)
    list_index_1 = [4,8,12,16,20]
    a = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                         min_tracking_confidence=0.5) as hands:
        while a.isOpened():
            ret,video = a.read()
            if not ret:
                print("read failed")
                break
            else:
                video = cv2.cvtColor(video,cv2.COLOR_BGR2RGB)
                video.flags.writeable = False
                results = hands.process(video)
                video.flags.writeable = True
                video = cv2.cvtColor(video,cv2.COLOR_RGB2BGR)
            list_landmark = []
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    
                    for id,lm in enumerate(hand_landmark.landmark):
                        h,w,c = video.shape
            #chuyển đổi tọa độ chuẩn hóa sang tọa độ pixel:
            #vì nếu muốn xử lí các điểm trên hình ảnh thì cần phải sử dụng tọa độ pixel
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        list_landmark.append([id,cx,cy])
                    mp_drawing.draw_landmarks(video,hand_landmark,
                                               connections = mp_hands.HAND_CONNECTIONS,
                                               landmark_drawing_spec = draw_spec_2,
                                               connection_drawing_spec = draw_spec_1)
                    b = []
                    if len(list_landmark) !=  0:
                        if list_landmark[list_index_1[0]][1] > list_landmark[list_index_1[0]-1][1]:
                            b.append(1)
                        else:
                            b.append(0)
                        for id in range(1,5):
                            if list_landmark[list_index_1[id]][2] < list_landmark[list_index_1[id]-2][2]:
                                b.append(1)
                            else:
                                b.append(0)
                    total = b.count(1)
                    print(total)

            cv2.imshow(" ",video)
            k = cv2.waitKey(1)
            if k == ord("q") or k == ord("Q"):
                break
    a.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    detection_hand_gestures()

