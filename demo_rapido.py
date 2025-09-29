# -*- coding: utf-8 -*-
"""
Entrenamiento Simplificado del Detector de Frutas
Versión rápida y sin errores para demostración
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Configurar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs

print("🍎 ENTRENAMIENTO RÁPIDO DE DETECTOR DE FRUTAS")
print("=" * 60)

# Cargar datos
def load_data(dataset_path):
    print("📂 Cargando dataset...")
    
    class_names = sorted([d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"📋 {len(class_names)} clases encontradas:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    images, labels = [], []
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        img_files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📁 {class_name}: {len(img_files)} imágenes")
        
        # Tomar solo las primeras 100 imágenes por clase para demo rápida
        for img_file in img_files[:100]:
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))  # Tamaño menor para demo
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype('float32') / 255.0
                    images.append(img)
                    labels.append(class_idx)
            except:
                continue
    
    X = np.array(images)
    y = keras.utils.to_categorical(labels, len(class_names))
    
    print(f"✅ Dataset cargado: {len(X)} imágenes")
    return X, y, class_names

# Crear modelo simple
def create_simple_model(num_classes):
    print("🧠 Creando modelo CNN simple...")
    
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Modelo creado")
    return model

# Entrenar modelo
def train_model(model, X, y):
    print("🏋️ Entrenando modelo...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Entrenamiento: {len(X_train)}, Prueba: {len(X_test)}")
    
    # Entrenar con menos épocas para demo
    history = model.fit(
        X_train, y_train,
        epochs=10,  # Solo 10 épocas para demo rápida
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluar
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📈 Precisión final: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    return model, history, X_test, y_test

# Función principal
def main():
    dataset_path = r"d:\Carrera - Ing. Sistemas\Ciclo VI\percepcion\Dataset_Filtrado"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset no encontrado en: {dataset_path}")
        return
    
    # Cargar datos
    X, y, class_names = load_data(dataset_path)
    
    # Crear modelo
    model = create_simple_model(len(class_names))
    
    # Entrenar
    trained_model, history, X_test, y_test = train_model(model, X, y)
    
    # Guardar modelo
    trained_model.save('modelo_frutas_demo.h5')
    with open('clases_frutas_demo.json', 'w') as f:
        json.dump(class_names, f)
    
    print("\n💾 Modelo guardado como: modelo_frutas_demo.h5")
    
    # Gráfica simple
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('entrenamiento_demo.png', dpi=150)
    plt.show()
    
    print("📊 Gráfica guardada como: entrenamiento_demo.png")
    print("\n🎉 ¡Entrenamiento completado exitosamente!")
    
    # Información de precios
    fruit_prices = {
        'Apple Red 1': 'S/. 3.50 por kg',
        'Banana 1': 'S/. 2.80 por kg',
        'Cantaloupe 2': 'S/. 5.20 por kg',
        'Cocos 1': 'S/. 4.00 por unidad',
        'Granadilla 1': 'S/. 8.50 por kg',
        'Kiwi 1': 'S/. 12.00 por kg',
        'Orange 1': 'S/. 3.00 por kg',
        'Passion Fruit 1': 'S/. 15.00 por kg',
        'Pear Forelle 1': 'S/. 7.50 por kg',
        'Pineapple 1': 'S/. 6.00 por unidad'
    }
    
    print("\n💰 PRECIOS DE FRUTAS DETECTADAS:")
    print("-" * 40)
    for fruit, price in fruit_prices.items():
        print(f"{fruit:.<25} {price}")
    
    print(f"\n🚀 PROYECTO COMPLETADO:")
    print(f"✅ Modelo entrenado desde cero")
    print(f"✅ {len(class_names)} tipos de frutas detectadas")
    print(f"✅ Base de datos de precios integrada")
    print(f"✅ Gráficas de rendimiento generadas")
    print(f"✅ Archivos guardados para uso futuro")

if __name__ == "__main__":
    main()