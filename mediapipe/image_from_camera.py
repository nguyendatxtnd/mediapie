import cv2

def test_cam():
    a = cv2.VideoCapture(0)

    while True:
        ret,image = a.read()
        if ret == False:
            print("read data not successed")
            break
        cv2.imshow("data from wedcam",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    a.release()
    cv2.destroyAllWindows()
if __name__=='__main__':
    test_cam()
