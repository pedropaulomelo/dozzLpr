from ultralytics import YOLO
import cv2
from util import get_car, read_license_plate
import csv
from datetime import datetime

# Load models
coco_model = YOLO('./models/yolov8s.pt')
license_plate_detector = YOLO('./models/lprModel.pt')

# Load a video
# video_path = './videos/vid2.mov'
video_path = 'rtsp://admin:leteb000@192.168.0.199:554/cam/realmonitor?channel=1&subtype=0'
# video_path = 'rtsp://admin:leteb000@10.216.164.120:554/cam/realmonitor?channel=1&subtype=0'
# video_path = 'rtsp://admin:leteb000@172.16.11.253:554/cam/realmonitor?channel=1&subtype=0'
cap = cv2.VideoCapture(video_path)

vehicles = [2, 3, 5, 7]

fps = cap.get(cv2.CAP_PROP_FPS)
# Calcular quantos frames pular para processar cerca de 6 frames por segundo
skip_frames = int(fps / 3)

frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    # print(f"Resolução do frame processado: {frame.shape[1]}x{frame.shape[0]}")

    # Processar apenas a cada 'skip_frames' frames
    if frame_counter % skip_frames == 0:
        results = {}

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            car_bbox = get_car([x1, y1, x2, y2], detections_)

            if car_bbox is not None:
                xcar1, ycar1, xcar2, ycar2 = car_bbox

                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results['car'] = {'bbox': [xcar1, ycar1, xcar2, ycar2],
                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                        'text': license_plate_text,
                                                        'bbox_score': score,
                                                        'text_score': license_plate_text_score}}

                    # Desenhar a caixa verde em torno do carro
                    cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)

                    # Desenhar a caixa vermelha em torno da placa de licença
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # Adicionar texto acima da placa de licença
                    cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 15)

    frame_counter += 1

    # Mostrar o frame (pode mostrar todos os frames ou apenas os processados)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()