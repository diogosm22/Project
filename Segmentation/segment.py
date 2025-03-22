import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO


# Carregar o modelo YOLO
model = YOLO("C:/Users/diogo/Desktop/python/yolov11/Segmentation/yolov11sseg.pt")

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
        break  # Encerra o loop se o usuário cancelar a seleção

    # Ler a imagem selecionada
    frame = cv2.imread(image_path)

    if frame is None:
        print("Erro ao carregar a imagem. Tente novamente.")
        continue  # Volta ao início do loop para selecionar outra imagem

    # Fazer a segmentação com o modelo YOLO
    results = model(frame)

    # Anotar e exibir a imagem segmentada
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv11 Segmentation", annotated_frame)

    # Aguardar interação do usuário
    print("Pressione 'Q' para carregar outra imagem ou 'K' para sair.")
    key = cv2.waitKey(0) & 0xFF  # Obtém apenas os 8 bits menos significativos

    if key == ord('q') or key == ord('Q'):  # Se "Q" for pressionado, permite carregar outra imagem
        cv2.destroyAllWindows()
        continue
    elif key == ord('k') or key == ord('K'):  # Se "K" for pressionado, encerra o programa
        print("Saindo do programa.")
        cv2.destroyAllWindows()
        break
