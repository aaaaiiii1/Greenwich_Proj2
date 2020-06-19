from __future__ import division
import cv2
import time
import sys
import numpy as np


# method để detect face dung Haar
def detectFaceOpenCVHaar(faceCascade, frame, inHeight=300, inWidth=0):
    # gán biến frameOpenCVHaar với cái frame mà camera hiện tại ghi hình
    frameOpenCVHaar = frame.copy()
    # set dài rộng cho biến bằng với dài rộng của frame camera
    frameHeight = frameOpenCVHaar.shape[0]
    frameWidth = frameOpenCVHaar.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    # đoạn ảnh có khuôn mặt trong inWidth và inHeight
    frameOpenCVHaarSmall = cv2.resize(frameOpenCVHaar, (inWidth, inHeight))
    # đổ ảnh lấy được sang ảnh gray
    frameGray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)

    # detect mặt từ đoạn ảnh gray ở trên bằng opencv, gán ảnh đó vô biến faces
    faces = faceCascade.detectMultiScale(frameGray)
    # tạo một array để chứa ảnh khuôn mặt
    bboxes = []

    # xác định chu vi của khuôn mặt trong faces - đã được detect ở trên
    for (x, y, w, h) in faces:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        # xác định phần ảnh có khuôn mặt, gán vào array cvRect
        cvRect = [int(x1 * scaleWidth), int(y1 * scaleHeight),
                  int(x2 * scaleWidth), int(y2 * scaleHeight)]
        # nhét ảnh khuôn mặt từ cvRect vào array vừa tạo ở trên
        bboxes.append(cvRect)
        # vẽ khung xanh xung quanh khuôn mặt trên cửa sổ ghi hình
        cv2.rectangle(frameOpenCVHaar, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                      int(round(frameHeight / 150)), 4)
    return frameOpenCVHaar, bboxes


if __name__ == "__main__":
    # vòng lập để không thể tắt màn hình video camera
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]
        # faceCascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    # faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier('D:\\FaceDetectionComparison\\haarcascade_frontalface_default.xml')

    # gán cái video hiện tại từ camera cho biến cap
    cap = cv2.VideoCapture(source)
    # quay phim từng frame một
    hasFrame, frame = cap.read()

    # định dạng file ghi ra là .avi, ở dưới sẽ dùng cái này
    vid_writer = cv2.VideoWriter('output-haar-{}.avi'.format(str(source).split(".")[0]),
                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame.shape[1], frame.shape[0]))

    # các biến để tính framerate
    frame_count = 0
    tt_opencvHaar = 0
    while (1):
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        frame_count += 1

        # tính framerate, số lượng frame/s
        t = time.time()
        # lấy dữ liệu từ return của detectFaceOpenCVHaar
        # rồi gán nó cho 2 biến outOpencvHaar và bboxes theo thứ tự return
        outOpencvHaar, bboxes = detectFaceOpenCVHaar(faceCascade, frame)
        tt_opencvHaar += time.time() - t
        fpsOpencvHaar = frame_count / tt_opencvHaar

        # thêm text vào màn hình video của camera
        label = "OpenCV Haar ; FPS : {:.2f}".format(fpsOpencvHaar)
        cv2.putText(outOpencvHaar, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        outGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray video", outGray)
        cv2.imshow("Face Detection Comparison", outOpencvHaar)

        # ghi video từ màn hình camera ra file .avi
        vid_writer.write(outOpencvHaar)
        if frame_count == 1:
            tt_opencvHaar = 0

        key = cv2.waitKey(1) & 0xFF
        # bấm Q để thoát
        if key == ord("q"):
            break
    # thoát tất cả khi xong
    cv2.destroyAllWindows()
    vid_writer.release()
