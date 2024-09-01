from ultralytics import YOLO
import cv2


# load yolov8 model
# model = YOLO('./models/lprModel.pt')
model = YOLO('.models/yolov8s.pt')

# load video
# video_path = './videos/test2.mov'
video_path = './videos/test1.mov'
# video_path = 'rtsp://admin:leteb000@10.216.104.61:554/cam/realmonitor?channel=6&subtype=0'
# video_path = 'rtsp://admin:leteb000@192.168.202.11:554/cam/realmonitor?channel=1&subtype=0'
# video_path = 'rtsp://admin:leteb000@192.168.1.100:554/cam/realmonitor?channel=5&subtype=0'

cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True)

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break