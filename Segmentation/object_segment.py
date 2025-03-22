# A usar object detection, mas a converter as bounding boxes para circulos, atribuir cor aos circulos e remover o fundo
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from ultralytics import YOLO

# Função para desenhar apenas o perímetro dos círculos ao redor das detecções
def draw_circle_perimeters(frame, detections):
    for x1, y1, x2, y2 in detections:
        # Calcular o centro e o raio do círculo
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        radius = int(max((x2 - x1), (y2 - y1)) / 2)   # Pequeno ajuste no raio

        # Desenhar apenas o contorno do círculo (interior preto, borda mais fina)
        cv2.circle(frame, center, radius, (255, 255, 255), 1, cv2.LINE_AA)  # Contorno branco, espessura 1

# Carregar o modelo YOLO
model = YOLO("C:/Users/diogo/Desktop/python/yolov11/yolov11s/modelo3.pt")
#model = YOLO("C:/Users/diogo/Desktop/python/yolov11/yolov11l/yolov11l_holes_2.pt")

# Função para abrir uma imagem usando um diálogo de arquivo
def load_image():
    root = tk.Tk()
    root.withdraw()  # Esconder a janela principal
    return filedialog.askopenfilename(title="Selecione uma imagem", filetypes=[("Arquivos de imagem", "*.jpg;*.png;*.jpeg")])

while True:
    # Carregar a imagem
    image_path = load_image()
    if not image_path:
        print("Nenhuma imagem selecionada. Saindo.")
        break

    # Ler a imagem selecionada
    frame = cv2.imread(image_path)
    if frame is None:
        print("Erro ao carregar a imagem. Tente novamente.")
        continue

    # Redimensionar a imagem (opcional)
    frame_resized = cv2.resize(frame, (640, 640))

    # Criar uma cópia para exibir a imagem original ao lado da segmentada
    original_image = frame_resized.copy()

    # Realizar a detecção de objetos com YOLO
    results = model.predict(frame_resized, conf=0.5, verbose=False, max_det=1000)

    # Lista para armazenar as detecções (coordenadas)
    detections = []

    # Extrair as coordenadas das bounding boxes
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Coordenadas da bounding box
        detections.append((x1, y1, x2, y2))  # Adicionar à lista de detecções

    # Criar uma imagem preta do mesmo tamanho da original
    segmented_image = np.zeros_like(frame_resized)

    # Desenhar apenas os contornos dos círculos
    draw_circle_perimeters(segmented_image, detections)

    # Concatenar as duas imagens lado a lado para exibição
    combined_image = np.hstack((original_image, segmented_image))

    # Exibir a imagem original e a segmentada lado a lado
    cv2.imshow("Original (Esquerda) | Segmentada (Direita)", combined_image)

    # Esperar por uma tecla pressionada
    print("Pressione 'Q' para carregar outra imagem ou 'K' para sair.")
    key = cv2.waitKey(0) & 0xFF  # Obtém apenas os 8 bits menos significativos

    if key == ord('q') or key == ord('Q'):  # Se "Q" for pressionado, permite carregar outra imagem
        cv2.destroyAllWindows()
        continue
    elif key == ord('k') or key == ord('K'):  # Se "K" for pressionado, encerra o programa
        print("Saindo do programa.")
        cv2.destroyAllWindows()
        break
