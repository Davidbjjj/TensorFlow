import cv2
import serial
import time
import numpy as np
import tensorflow as tf
from pygame import mixer

# Inicializa o som
mixer.init()
try:
    mixer.music.load('alert.mp3')
except:
    print("⚠️ Arquivo de alerta 'alert.mp3' não encontrado.")

# Labels
labels = ["Pedestre", "Sem Pedestre", "Carro"]
alert_threshold_cm = 20

# Carrega modelo
model = tf.keras.models.load_model('model.h5')  # ajuste se for .json
print("✅ Modelo carregado.")

# Serial
ser = serial.Serial('COM3', 9600)  # troque por '/dev/ttyUSB0' no Linux

# Tempo para alertas
last_alert_time = 0

# Função para capturar e processar imagem da webcam
def capture_and_predict(frame):
    img = cv2.resize(frame, (96, 96))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    input_tensor = gray.reshape(1, 96, 96, 1).astype(np.float32) / 255.0
    preds = model.predict(input_tensor)[0]
    max_idx = np.argmax(preds)
    return labels[max_idx], preds[max_idx] * 100

# Loop principal
cap = cv2.VideoCapture(0)
print("🎥 Webcam iniciada.")

while True:
    # Lê distância da serial
    try:
        line = ser.readline().decode('utf-8').strip()
        if "Distância:" in line:
            distancia = float(line.split(":")[1].replace("cm", "").strip())
        else:
            continue
    except Exception as e:
        print("Erro na leitura serial:", e)
        continue

    # Captura frame
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem.")
        break

    # Classifica
    classe, confianca = capture_and_predict(frame)

    # Exibe info
    texto = f"{classe} - {distancia:.1f}cm - Confiança: {confianca:.2f}%"
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Verifica condição de alerta
    now = time.time()
    if classe == "Pedestre" and distancia < alert_threshold_cm:
        if now - last_alert_time > 2:
            print("⚠️ ALERTA: Pedestre próximo!")
            mixer.music.play()
            last_alert_time = now
        cv2.putText(frame, "🚨 ALERTA!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Mostra na tela
    cv2.imshow("Sistema de Monitoramento", frame)

    # Tecla ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
ser.close()
