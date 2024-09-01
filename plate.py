import cv2
from ultralytics import YOLO
import util  # Importando util.py
import numpy as np

# Carregar o modelo YOLO
model = YOLO('./models/lprModel.pt')

# Caminho do vídeo
# video_path = 'rtsp://admin:leteb000@172.16.11.253:554/cam/realmonitor?channel=1&subtype=0'
video_path = 'rtsp://admin:leteb000@192.168.0.199:554/cam/realmonitor?channel=1&subtype=0'
# video_path = 'rtsp://admin:leteb000@10.216.164.120:554/cam/realmonitor?channel=1&subtype=0'
# video_path = './videos/vid2.mov'

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
    
    # Processar apenas a cada 'skip_frames' frames
    if frame_count % skip_frames == 0:
        results = model.track(frame, persist=True)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            license_plate_crop = frame[y1:y2, x1:x2]

            # Ler a placa de licença
            license_text, confidence = util.read_license_plate(license_plate_crop)
            print(license_text, confidence)
            
            # Verificar se a confiança é maior que 0.1 e exibir o texto
            if license_text is not None and confidence > 0.1:
                # Configuração para o texto
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3.0  # Ajuste conforme necessário
                font_color = (0, 0, 255)  # Vermelho
                font_thickness = 7  # Negrito

                # Posição para exibir o texto (acima da caixa delimitadora)
                text_position = (x1 - 20, y1 - 100)  # Ajuste o -10 se necessário

                # Desenhar o texto na imagem
                cv2.putText(frame, license_text, text_position, font, font_scale, font_color, font_thickness)

                # Desenhar o box ao redor da placa
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 10)

                # Expanda a área ao redor da caixa da placa para detectar a cor do carro
                padding = 50  # Número de pixels para expandir ao redor da caixa
                x1_exp = max(x1 - padding, 0)
                y1_exp = max(y1 - padding, 0)
                x2_exp = min(x2 + padding, frame.shape[1])
                y2_exp = min(y2 + padding, frame.shape[0])

                # Extrair a região expandida ao redor da placa
                car_region = frame[y1_exp:y2_exp, x1_exp:x2_exp]

                # Calcular a cor média na região expandida
                avg_color = np.mean(car_region, axis=(0, 1))
                avg_color_bgr = tuple(map(int, avg_color))

                # Desenhar um retângulo com a cor média ao redor da caixa da placa
                cv2.rectangle(frame, (x1_exp, y1_exp), (x2_exp, y2_exp), avg_color_bgr, 10)

                # Desenhar o retângulo do corte da placa
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)  # Retângulo Azul

                # # Desenhar o retângulo da área expandida
                # cv2.rectangle(frame, (x1_exp, y1_exp), (x2_exp, y2_exp), (0, 255, 255), 5)  # Retângulo Amarelo

                # Mostrar a cor detectada
                print(f"Cor média do carro (BGR): {avg_color_bgr}")

        frame_ = results[0].plot()
        cv2.imshow('frame', frame)

    # Aguardar 1/60 segundos
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()