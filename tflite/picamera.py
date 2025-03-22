import argparse

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2, Preview

# Argumentos da linha de comando
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Caminho para o modelo .tflite")
parser.add_argument("--labels", required=False, help="Caminho para o arquivo de labels")
args = parser.parse_args()

# Carregar o modelo TFLite
interpreter = tflite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()

# Obter detalhes dos tensores de entrada e saída
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Carregar labels se fornecido
labels = []
if args.labels:
    with open(args.labels, "r") as f:
        labels = [line.strip() for line in f.readlines()]

# Função para processar a imagem e fazer a inferência
def process_image(image):
    # Redimensionar para o tamanho de entrada do modelo
    input_size = (input_shape[1], input_shape[2])
    image_resized = cv2.resize(image, input_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_input = np.expand_dims(image_rgb, axis=0).astype(np.float32) / 255.0  # Normalização

    # Fazer a inferência
    interpreter.set_tensor(input_details[0]['index'], image_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Processar as detecções
    detections = output_data[0]  # Remover a dimensão do batch
    for detection in detections.T:  # Transpor para iterar sobre as detecções
        x, y, w, h, confidence = detection

        # Filtre detecções com baixa confiança
        if confidence > 0.5:  # Ajuste o limiar de confiança conforme necessário
            # Desnormalize as coordenadas para o tamanho da imagem original
            x = int(x * image.shape[1])
            y = int(y * image.shape[0])
            w = int(w * image.shape[1])
            h = int(h * image.shape[0])

            # Desenhe a bounding box
            label = f"Object {confidence:.2f}" if not labels else labels[int(detection[4])]
            cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
            cv2.putText(image, label, (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Inicializar a Picamera2
picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

while True:
    # Capturar um frame da Picamera
    frame = picam2.capture_array("main")

    # Processar o frame
    output_frame = process_image(frame)

    # Mostrar o frame processado
    cv2.imshow("Detecção", output_frame)

    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
picam2.stop()
cv2.destroyAllWindows()