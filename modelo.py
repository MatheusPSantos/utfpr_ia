# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Importações

import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import PIL
from sklearn import preprocessing, model_selection, metrics
import numpy as np

# %% [markdown]
# Adicionando as configurações de diretório e outros artefatos para o modelo

# %%
# diretorio da localização das imagens de treino
DIR = r"C:\Users\perei\Desktop\universidade\inteligencia artificial\utfpr_ia"
CATEGORIAS = ['mask', 'no_mask']

# iniciando os arrays de Data e labels vazios
data = []
labels = []

# %% [markdown]
# Adicionando as imagens de treino

# %%
for categoria in CATEGORIAS:
    path = os.path.join(DIR, categoria)
    for imagem in os.listdir(path):
        image_path = os.path.join(path, imagem)
        image = keras.preprocessing.image.load_img(image_path, target_size=(224,224))
        image = keras.preprocessing.image.img_to_array(image)
        image = keras.applications.mobilenet_v2.preprocess_input(image)

        data.append(image)
        labels.append(categoria)

# %% [markdown]
# Transformando valores categóricos em vetores binários esparso

# %%
bin_labels = preprocessing.LabelBinarizer()
labels = bin_labels.fit_transform(labels)
labels = keras.utils.to_categorical(labels)

# %% [markdown]
# Convertendo os arrays de data e labels para o formato float32 com o numpy

# %%
data = np.array(data, dtype="float32")
labels = np.array(labels)

# %% [markdown]
# Splitando o array em subset de testes e trainamento

# %%
(trainX, testX, trainY, testY) = model_selection.train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# %% [markdown]
# Gerador de imagem de treinamento

# %%
aumento = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.15, horizontal_flip=True,
    fill_mode="nearest"
)

# %% [markdown]
# Carregamento da rede MobileNetV2

# %%
modeloBase = keras.applications.MobileNetV2(
    weights="imagenet", include_top=False, input_tensor=keras.layers.Input(shape=(224, 224, 3))
)

cabecalhoModelo = modeloBase.output
cabecalhoModelo = keras.layers.AveragePooling2D(pool_size=(7,7))(cabecalhoModelo)
cabecalhoModelo = keras.layers.Flatten(name="flatten")(cabecalhoModelo)
cabecalhoModelo = keras.layers.Dense(128, activation="relu")(cabecalhoModelo)
cabecalhoModelo = keras.layers.Dropout(0.5)(cabecalhoModelo)
cabecalhoModelo = keras.layers.Dense(2, activation="softmax")(cabecalhoModelo)

# %% [markdown]
# Criando o modelo de treinamento

# %%
modelo = keras.Model(inputs=modeloBase.input, outputs=cabecalhoModelo)

# %% [markdown]
# Fazendo um loop em cima das camadas do modelo base

# %%
for camada in modeloBase.layers:
    camada.trainable = False

# %% [markdown]
# Compilando o nosso modelo

# %%
# algumas definições
EPOCAS = 20

# definindo o otimizador
otimizador = keras.optimizers.Adam(learning_rate=1e-4, decay=1e-4/EPOCAS)

modelo.compile(loss="binary_crossentropy", optimizer=otimizador, metrics=['accuracy'])

# %% [markdown]
# Treinando o cabeçalho da rede

# %%
head = modelo.fit(
    aumento.flow(trainX, trainY, batch_size=32),
    steps_per_epoch=len(trainX) // 32,
    validation_data=(testX, testY),
    validation_steps=len(testX) // 32,
    epochs=EPOCAS
)

# %% [markdown]
# Realizando teste no modelo salvo

# %%
predIdxs = modelo.predict(testX, batch_size=32)

# %% [markdown]
# Para cada imagem no set de teste, precisamos do indice da label com mais probabilidade de predição

# %%
predIdxs = np.argmax(predIdxs, axis=1)

# %% [markdown]
# Report de classificação

# %%
print(metrics.classification_report(testY.argmax(axis=1), predIdxs, target_names=bin_labels.classes_))

# %% [markdown]
# Salvando o modelo

# %%
modelo.save("modelo_reconhecimento_mascara.h5")

# %% [markdown]
# Gráfico de acurácia e perda do modelo

# %%
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,EPOCAS), head.history["loss"], label="perda treino")
plt.plot(np.arange(0,EPOCAS), head.history["val_loss"], label="perca validação")
plt.plot(np.arange(0,EPOCAS), head.history["accuracy"], label="acurácia treino")
plt.plot(np.arange(0,EPOCAS), head.history["val_accuracy"], label="acurácia validação")
plt.title("Perca e acurácia de treino")
plt.ylabel("perca/acurácia")
plt.legend(loc="lower left")
plt.xlabel("Época")


