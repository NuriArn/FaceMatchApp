import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Directorios del dataset
train_dir = "C:/Users/victoria.spandonari/Documents/Ciencia de Datos/MODELIZADO/Famous People Faces/entrenamiento"
val_dir = "C:/Users/victoria.spandonari/Documents/Ciencia de Datos/MODELIZADO/Famous People Faces/prueba"

# Preprocesamiento de las imágenes
train_data_gen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
val_data_gen = ImageDataGenerator(rescale=1/255.0)

train_images = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=2021
)

val_images = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=2021
)

test_images = val_data_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

from tensorflow.keras.optimizers import Adam
# Cargar el modelo preentrenado MobileNetV2 sin las capas superiores
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar las capas del modelo base
base_model.trainable = False

# Crear la arquitectura de la red neuronal
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # Evitar sobreajuste
predictions = Dense(train_images.num_classes, activation='softmax')(x)

# Definir el modelo completo
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001),  # Cambia lr por learning_rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
epochs = 30
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=epochs,
    steps_per_epoch=train_images.samples // train_images.batch_size,
    validation_steps=val_images.samples // val_images.batch_size
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_images)
print(f"Test accuracy: {test_acc}")
