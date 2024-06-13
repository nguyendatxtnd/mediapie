import cv2
import mediapipe as mp

def face_mesh_2():
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    draw_spec = mp_drawing.DrawingSpec(color = (0, 128, 0),thickness = 1, circle_radius = 1)
    a = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(static_image_mode = False,
                               max_num_faces = 2,
                               min_detection_confidence = 0.5) as face_mesh:
        while a.isOpened():
            ret,image = a.read()
            if ret == False:
                print("read image not sucessed")
                break
            else:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for mesh in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image,mesh,
                                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                                  landmark_drawing_spec=draw_spec,
                                                  connection_drawing_spec=draw_spec)
            cv2.imshow(" ",image)
            k = cv2.waitKey(1)
            if k == ord("q") or k == ord("Q"):
                break
    a.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    face_mesh_2()