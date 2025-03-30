import cv2

# Carregar modelo de super-resolução do OpenCV
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("ESPCN_x4.pb")  # Use ESPCN, FSRCNN ou EDSR
sr.setModel("espcn", 4)  # Modelo ESPCN com fator 4x

# Carregar imagem
imagem = cv2.imread("target/img.jpeg")

# Aplicar super-resolução
imagem_sr = sr.upsample(imagem)

# Salvar resultado
cv2.imwrite("output.jpg", imagem_sr)
