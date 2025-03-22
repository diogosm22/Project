import argparse

import cv2
import numpy as np
import tensorflow.lite as tflite

# Argumentos da linha de comando
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Caminho para o modelo .tflite")
parser.add_argument("--image", required=False, help="Caminho para uma imagem (opcional)")
args = parser.parse_args()

# Carregar o modelo TFLite
interpreter = tflite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()

# Obter detalhes dos tensores de entrada e saída
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

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
            label = f"Object {confidence:.2f}"
            cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
            cv2.putText(image, label, (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Testar com uma imagem
if args.image:
    image = cv2.imread(args.image)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem {args.image}")
    else:
        output_image = process_image(image)
        cv2.imshow("Det", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    # Testar com a webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = process_image(frame)
        cv2.imshow("Det", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()