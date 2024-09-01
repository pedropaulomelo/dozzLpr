import cv2
from ultralytics import YOLO
import util  # Importando util.py

# Carregar o modelo YOLO
model = YOLO('./models/lprModel.pt')

# Caminho do vídeo
# video_path = 'rtsp://admin:leteb000@10.216.104.61:544/cam/realmonitor?channel=6&subtype=0'
# video_path = 'rtsp://admin:leteb000@172.16.11.4:554/cam/realmonitor?channel=18&subtype=0'
# video_path = 'rtsp://admin:leteb000@192.168.1.100:554/cam/realmonitor?channel=5&subtype=0'
video_path = 'rtsp://admin:leteb000@172.16.11.253:554/cam/realmonitor?channel=1&subtype=0'
# video_path = './videos/test4.mov'

cap = cv2.VideoCapture(video_path)

# Verificar se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo")
    exit()

# Obter a taxa de frames do vídeo
fps = cap.get(cv2.CAP_PROP_FPS)

# Calcular o número de frames para pular para analisar aproximadamente 6 frames por segundo
skip_frames = int(fps / 6)

# Contador para acompanhar o número atual de frames
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    # print(f"Resolução do frame processado: {frame.shape[1]}x{frame.shape[0]}")

    # Processar apenas a cada 'skip_frames' frames
    if frame_count % skip_frames == 0:
        results = model.track(frame, persist=True)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            license_plate_crop = frame[y1:y2, x1:x2]

            # Ler a placa de licença
            license_text, confidence = util.read_license_plate(license_plate_crop)
            print(license_text, confidence)
            # Verificar se a confiança é maior que 0.7 e exibir o texto
            if license_text is not None and confidence > 0.1:
                # Configuração para o texto
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 7.0  # Ajuste conforme necessário
                font_color = (0, 0, 255)  # Vermelho
                font_thickness = 30  # Negrito

                # Posição para exibir o texto (acima da caixa delimitadora)
                text_position = (x1, y1 - 750)  # Ajuste o -10 se necessário

                # Desenhar o texto na imagem
                cv2.putText(frame, license_text, text_position, font, font_scale, font_color, font_thickness)

                license_text, confidence = util.read_license_plate(license_plate_crop)
        frame_ = results[0].plot()
        cv2.imshow('frame', frame)

    # Aguardar 1/60 segundos
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
