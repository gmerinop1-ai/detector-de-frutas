# -*- coding: utf-8 -*-
"""
Detector Simple - Cargar imagen por ruta
Ejemplo de cómo cargar y analizar una imagen específica
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import json

# Configurar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cargar_modelo():
    """Carga el modelo entrenado"""
    try:
        if os.path.exists('modelo_frutas_demo.h5'):
            model = keras.models.load_model('modelo_frutas_demo.h5')
            with open('clases_frutas_demo.json', 'r') as f:
                class_names = json.load(f)
            print("✅ Modelo cargado exitosamente")
            return model, class_names
        else:
            print("❌ No se encontró el modelo. Ejecuta primero: python demo_rapido.py")
            return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def detectar_fruta(ruta_imagen):
    """
    Detecta qué fruta hay en una imagen
    
    Args:
        ruta_imagen (str): Ruta completa a la imagen
    
    Returns:
        dict: Información de la predicción
    """
    
    # Cargar modelo
    model, class_names = cargar_modelo()
    if model is None:
        return None
    
    # Precios de las frutas
    precios = {
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
    
    try:
        # 1. Cargar imagen
        print(f"📸 Cargando imagen: {ruta_imagen}")
        img = cv2.imread(ruta_imagen)
        
        if img is None:
            print("❌ No se pudo cargar la imagen")
            return None
        
        # 2. Preprocesar imagen
        img_resized = cv2.resize(img, (64, 64))  # Tamaño del modelo demo
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # 3. Hacer predicción
        print("🧠 Analizando imagen...")
        predictions = model.predict(img_batch, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = class_names[predicted_idx]
        
        # 4. Preparar resultado
        resultado = {
            'fruta': predicted_class,
            'confianza': confidence,
            'precio': precios.get(predicted_class, 'Precio no disponible'),
            'imagen': img_rgb,
            'todas_predicciones': predictions[0]
        }
        
        return resultado
        
    except Exception as e:
        print(f"❌ Error procesando imagen: {e}")
        return None

def mostrar_resultado(resultado):
    """Muestra el resultado de la detección"""
    if resultado is None:
        return
    
    print(f"\n🍎 RESULTADO:")
    print(f"   Fruta detectada: {resultado['fruta']}")
    print(f"   Confianza: {resultado['confianza']:.2%}")
    print(f"   💰 Precio: {resultado['precio']}")
    
    # Mostrar imagen con resultado
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(resultado['imagen'])
    plt.title(f"Fruta: {resultado['fruta']}\nConfianza: {resultado['confianza']:.2%}")
    plt.axis('off')
    
    # Mostrar top 3 predicciones
    plt.subplot(1, 2, 2)
    top_3_idx = np.argsort(resultado['todas_predicciones'])[-3:][::-1]
    
    # Cargar nombres de clases
    _, class_names = cargar_modelo()
    if class_names:
        top_3_names = [class_names[i] for i in top_3_idx]
        top_3_probs = resultado['todas_predicciones'][top_3_idx]
        
        colors = ['green', 'blue', 'orange']
        bars = plt.bar(range(3), top_3_probs, color=colors, alpha=0.7)
        plt.xticks(range(3), [name.replace(' ', '\n') for name in top_3_names], rotation=0)
        plt.ylabel('Probabilidad')
        plt.title('Top 3 Predicciones')
        
        # Añadir valores
        for bar, prob in zip(bars, top_3_probs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# EJEMPLOS DE USO:

def ejemplo_basico():
    """Ejemplo básico de uso"""
    print("🍎 EJEMPLO BÁSICO DE DETECCIÓN")
    print("=" * 40)
    
    # Puedes cambiar esta ruta por cualquier imagen
    ruta = input("📂 Ingresa la ruta de la imagen: ").strip().replace('"', '')
    
    if not os.path.exists(ruta):
        print("❌ La imagen no existe")
        return
    
    # Detectar fruta
    resultado = detectar_fruta(ruta)
    
    # Mostrar resultado
    mostrar_resultado(resultado)

def ejemplo_con_imagen_del_dataset():
    """Ejemplo usando una imagen del dataset"""
    print("🍎 EJEMPLO CON IMAGEN DEL DATASET")
    print("=" * 40)
    
    dataset_path = r"d:\Carrera - Ing. Sistemas\Ciclo VI\percepcion\Dataset_Filtrado"
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset no encontrado")
        return
    
    # Mostrar clases disponibles
    clases = os.listdir(dataset_path)
    print("📋 Clases disponibles:")
    for i, clase in enumerate(clases):
        print(f"  {i+1}. {clase}")
    
    try:
        seleccion = int(input(f"\n👉 Selecciona una clase (1-{len(clases)}): ")) - 1
        if 0 <= seleccion < len(clases):
            clase_seleccionada = clases[seleccion]
            clase_path = os.path.join(dataset_path, clase_seleccionada)
            
            # Obtener primera imagen de la clase
            imagenes = [f for f in os.listdir(clase_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if imagenes:
                imagen_path = os.path.join(clase_path, imagenes[0])
                print(f"📸 Probando con: {imagenes[0]}")
                
                # Detectar
                resultado = detectar_fruta(imagen_path)
                mostrar_resultado(resultado)
            else:
                print("❌ No se encontraron imágenes en esa clase")
        else:
            print("❌ Selección no válida")
    except ValueError:
        print("❌ Ingresa un número válido")

if __name__ == "__main__":
    print("🍎 DETECTOR DE FRUTAS - CARGAR IMAGEN")
    print("=" * 50)
    
    while True:
        print(f"\n📋 OPCIONES:")
        print(f"1. 📂 Cargar imagen por ruta")
        print(f"2. 🎲 Probar con imagen del dataset")
        print(f"3. ❌ Salir")
        
        opcion = input(f"\n👉 Selecciona (1-3): ").strip()
        
        if opcion == "1":
            ejemplo_basico()
        elif opcion == "2":
            ejemplo_con_imagen_del_dataset()
        elif opcion == "3":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida")