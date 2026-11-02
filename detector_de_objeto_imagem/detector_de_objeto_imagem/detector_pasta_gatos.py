import os
import cv2
from ultralytics import YOLO

print("Carregando o modelo YOLOv8...")
model = YOLO("yolov8n.pt")

pasta_origem = "cat_images"
pasta_destino = "resultados_gatos"

os.makedirs(pasta_destino, exist_ok=True)

extensoes_validas = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

if not os.path.exists(pasta_origem):
    print(f"❌ Erro: A pasta '{pasta_origem}' não foi encontrada!")
    print(f"Por favor, crie a pasta '{pasta_origem}' e coloque as imagens dentro dela.")
    exit()

arquivos = [f for f in os.listdir(pasta_origem) 
if f.lower().endswith(extensoes_validas)]

if not arquivos:
        print(f"⚠️ Nenhuma imagem válida encontrada na pasta '{pasta_origem}'.")
        exit()

print(f"📸 Encontradas {len(arquivos)} imagens para processar em '{pasta_origem}'...\n")

ID_GATO = 15  # ID numérico da classe 'cat' no dataset COCO
total_gatos_detectados = 0

for i, nome_arquivo in enumerate(arquivos, start=1):
    caminho_imagem = os.path.join(pasta_origem, nome_arquivo)
    print(f"[{i}/{len(arquivos)}] Analisando: {nome_arquivo}...")
    # Executa a inferência de Visão Computacional
    resultados = model(caminho_imagem, verbose=False)
    resultado = resultados[0]

    gatos_na_foto = 0

	# Analisa as caixas delimitadoras encontradas
    for box in resultado.boxes:
        classe_id = int(box.cls[0])
        confianca = float(box.conf[0])
	
        if classe_id == ID_GATO:
            gatos_na_foto += 1
            total_gatos_detectados += 1
            print(f"  🐱 Gato detectado! Confiança: {confianca * 100:.1f}%")
	
if gatos_na_foto == 0:
	print("  ⚠️ Nenhum gato detectado nesta foto.")
	
	# Desenha as marcações na imagem
imagem_desenhada = resultado.plot()
	
	# Salva a imagem processada na pasta de destino
caminho_saida = os.path.join(pasta_destino, f"resultado_{nome_arquivo}")
cv2.imwrite(caminho_saida, imagem_desenhada)
print(f"  ✅ Salvo em: {caminho_saida}\n")
	
print("=" * 50)
print(f"🎉 Processamento concluído!")
print(f"📊 Total de fotos analisadas: {len(arquivos)}")
print(f"🐱 Total de gatos identificados: {total_gatos_detectados}")
print(f"📁 Imagens marcadas salvas na pasta: '{pasta_destino}'")