
import os
import cv2
from ultralytics import YOLO


# -------------------------------------------------------------
# 1. CARREGAMENTO DO MODELO DE INTELIGÊNCIA ARTIFICIAL
# -------------------------------------------------------------

print("Inicializando o detector de imagens...")

detector = YOLO("yolov8n.pt")


# -------------------------------------------------------------
# 2. CONFIGURAÇÃO DAS PASTAS
# -------------------------------------------------------------

diretorio_entrada = "cat_images"
diretorio_saida = "resultados_gatos"

os.makedirs(diretorio_saida, exist_ok=True)

formatos_permitidos = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
)


# -------------------------------------------------------------
# 3. VERIFICAÇÃO DOS ARQUIVOS
# -------------------------------------------------------------

if not os.path.isdir(diretorio_entrada):

    print(f"Erro: Diretório '{diretorio_entrada}' não encontrado!")
    print("Crie a pasta e adicione as imagens para continuar.")

    exit()


lista_imagens = [
    arquivo
    for arquivo in os.listdir(diretorio_entrada)
    if arquivo.lower().endswith(formatos_permitidos)
]


if len(lista_imagens) == 0:

    print("Nenhuma imagem compatível foi localizada.")

    exit()


print(
    f"\nTotal de imagens encontradas: {len(lista_imagens)}"
)


# -------------------------------------------------------------
# 4. CONFIGURAÇÃO DA DETECÇÃO
# -------------------------------------------------------------

CLASSE_GATO = 15

quantidade_gatos = 0


# -------------------------------------------------------------
# 5. ANÁLISE E IDENTIFICAÇÃO DOS GATOS
# -------------------------------------------------------------

for posicao, arquivo in enumerate(lista_imagens, start=1):

    caminho_atual = os.path.join(
        diretorio_entrada, arquivo
    )

    print(
        f"\n[{posicao}/{len(lista_imagens)}] "
        f"Verificando imagem: {arquivo}"
    )

    # Executa a detecção utilizando YOLO
    previsoes = detector(
        caminho_atual,
        verbose=False
    )

    deteccao = previsoes[0]

    gatos_encontrados = 0

    # Verifica os objetos identificados pelo modelo
    for objeto in deteccao.boxes:

        identificador = int(objeto.cls[0])
        nivel_confianca = float(objeto.conf[0])

        if identificador == CLASSE_GATO:

            gatos_encontrados += 1
            quantidade_gatos += 1

            print(
                f"  Gato identificado! "
                f"Precisão estimada: {nivel_confianca * 100:.1f}%"
            )

    # Informa quando nenhum gato é encontrado
    if gatos_encontrados == 0:

        print("  Nenhum gato foi identificado nesta imagem.")

    else:

        print(
            f"  Quantidade de gatos nesta imagem: "
            f"{gatos_encontrados}"
        )

    # Cria uma versão da imagem com as marcações
    imagem_resultado = deteccao.plot()

    # Define o caminho para salvar o resultado
    nome_saida = f"resultado_{arquivo}"

    destino_imagem = os.path.join(
        diretorio_saida, nome_saida
    )

    # Exporta a imagem processada
    sucesso = cv2.imwrite(
        destino_imagem,
        imagem_resultado
    )

    if sucesso:
        print(f"  Imagem salva: {destino_imagem}")
    else:
        print("  Não foi possível salvar esta imagem.")


# -------------------------------------------------------------
# 6. RELATÓRIO FINAL DO PROCESSAMENTO
# -------------------------------------------------------------

print("\n" + "=" * 50)

print("PROCESSAMENTO FINALIZADO!")

print(f"Imagens analisadas: {len(lista_imagens)}")

print(f"Gatos identificados: {quantidade_gatos}")

print(f"Resultados armazenados em: '{diretorio_saida}'")

print("=" * 50)
