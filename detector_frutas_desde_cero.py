# -*- coding: utf-8 -*-
"""
Detector de Frutas con Precios - Entrenamiento desde Cero
Proyecto de Percepción Computacional
Entrena un modelo CNN personalizado para detectar frutas y mostrar precios
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar TensorFlow para usar GPU si está disponible
print("🔧 Configurando TensorFlow...")
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"✅ GPU disponible: {physical_devices[0]}")
else:
    print("⚠️ Usando CPU (el entrenamiento será más lento)")

class FruitPriceDetector:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Base de datos de precios (simulada - en proyecto real conectarías a API)
        self.fruit_prices = {
            'Apple Red 1': {'price': 3.50, 'unit': 'kg', 'currency': 'S/.'},
            'Banana 1': {'price': 2.80, 'unit': 'kg', 'currency': 'S/.'},
            'Cantaloupe 2': {'price': 5.20, 'unit': 'kg', 'currency': 'S/.'},
            'Cocos 1': {'price': 4.00, 'unit': 'unidad', 'currency': 'S/.'},
            'Granadilla 1': {'price': 8.50, 'unit': 'kg', 'currency': 'S/.'},
            'Kiwi 1': {'price': 12.00, 'unit': 'kg', 'currency': 'S/.'},
            'Orange 1': {'price': 3.00, 'unit': 'kg', 'currency': 'S/.'},
            'Passion Fruit 1': {'price': 15.00, 'unit': 'kg', 'currency': 'S/.'},
            'Pear Forelle 1': {'price': 7.50, 'unit': 'kg', 'currency': 'S/.'},
            'Pineapple 1': {'price': 6.00, 'unit': 'unidad', 'currency': 'S/.'}
        }
    
    def load_and_preprocess_data(self, dataset_path, img_size=(128, 128)):
        """
        Carga y preprocesa el dataset completo
        """
        print("📂 Cargando dataset desde:", dataset_path)
        
        if not os.path.exists(dataset_path):
            print(f"❌ Error: No se encontró el dataset en {dataset_path}")
            return False
        
        # Obtener nombres de clases
        self.class_names = sorted([d for d in os.listdir(dataset_path) 
                                  if os.path.isdir(os.path.join(dataset_path, d))])
        
        print(f"📋 Clases encontradas ({len(self.class_names)}):")
        for i, class_name in enumerate(self.class_names):
            class_path = os.path.join(dataset_path, class_name)
            num_images = len([f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {i:2d}. {class_name}: {num_images} imágenes")
        
        # Cargar imágenes y etiquetas
        images = []
        labels = []
        
        print("\n🔄 Procesando imágenes...")
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"📁 Procesando {class_name}... ({len(image_files)} imágenes)")
            
            for i, img_file in enumerate(image_files):
                if i % 100 == 0:
                    print(f"   Procesadas: {i}/{len(image_files)}")
                
                img_path = os.path.join(class_path, img_file)
                try:
                    # Cargar imagen
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Redimensionar
                        img = cv2.resize(img, img_size)
                        # Convertir BGR a RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Normalizar píxeles (0-1)
                        img = img.astype('float32') / 255.0
                        
                        images.append(img)
                        labels.append(class_idx)
                        
                except Exception as e:
                    print(f"   ⚠️ Error procesando {img_file}: {e}")
        
        # Convertir a arrays numpy
        X = np.array(images)
        y = np.array(labels)
        
        print(f"\n✅ Dataset cargado exitosamente:")
        print(f"   📊 Total de imágenes: {len(X)}")
        print(f"   📐 Forma de imágenes: {X.shape}")
        print(f"   🏷️ Número de clases: {len(self.class_names)}")
        
        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Convertir etiquetas a categóricas
        self.y_train = keras.utils.to_categorical(self.y_train, len(self.class_names))
        self.y_test = keras.utils.to_categorical(self.y_test, len(self.class_names))
        
        print(f"📊 División del dataset:")
        print(f"   🏋️ Entrenamiento: {len(self.X_train)} imágenes")
        print(f"   🧪 Prueba: {len(self.X_test)} imágenes")
        
        return True
    
    def create_cnn_model(self, img_size=(128, 128)):
        """
        Crea la arquitectura CNN desde cero
        """
        print("🧠 Creando arquitectura CNN personalizada...")
        
        model = keras.Sequential([
            # Primera capa convolucional
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(*img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Segunda capa convolucional
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Tercera capa convolucional
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Cuarta capa convolucional
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Aplanar para capas densas
            layers.Flatten(),
            
            # Capas densas
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Capa de salida
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        self.model = model
        
        print("✅ Modelo creado exitosamente")
        print("\n📋 Resumen de la arquitectura:")
        model.summary()
        
        return model
    
    def train_model(self, epochs=30, batch_size=32):
        """
        Entrena el modelo CNN desde cero
        """
        if self.model is None:
            print("❌ Error: Modelo no creado. Ejecuta create_cnn_model() primero.")
            return None
        
        if self.X_train is None:
            print("❌ Error: Datos no cargados. Ejecuta load_and_preprocess_data() primero.")
            return None
        
        print(f"🚀 Iniciando entrenamiento del modelo...")
        print(f"   📊 Épocas: {epochs}")
        print(f"   📦 Tamaño de lote: {batch_size}")
        print(f"   🖼️ Imágenes de entrenamiento: {len(self.X_train)}")
        print(f"   🧪 Imágenes de validación: {len(self.X_test)}")
        
        # Callbacks para mejorar el entrenamiento
        callbacks = [
            # Parar si no mejora
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reducir learning rate si se estanca
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            # Guardar mejor modelo
            keras.callbacks.ModelCheckpoint(
                'mejor_modelo_frutas.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar modelo
        print("\n🏋️ Comenzando entrenamiento...")
        start_time = datetime.now()
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"\n✅ Entrenamiento completado en: {training_time}")
        
        # Guardar modelo final
        self.model.save('modelo_frutas_final.h5')
        
        # Guardar nombres de clases
        with open('clases_frutas.json', 'w', encoding='utf-8') as f:
            json.dump(self.class_names, f, ensure_ascii=False, indent=2)
        
        # Guardar historial
        with open('historial_entrenamiento.json', 'w') as f:
            history_dict = {
                'loss': [float(x) for x in self.history.history['loss']],
                'accuracy': [float(x) for x in self.history.history['accuracy']],
                'val_loss': [float(x) for x in self.history.history['val_loss']],
                'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
                'epochs': len(self.history.history['loss']),
                'training_time': str(training_time)
            }
            json.dump(history_dict, f, indent=2)
        
        print(f"💾 Archivos guardados:")
        print(f"   📁 modelo_frutas_final.h5")
        print(f"   📁 mejor_modelo_frutas.h5")
        print(f"   📁 clases_frutas.json")
        print(f"   📁 historial_entrenamiento.json")
        
        return self.history
    
    def evaluate_model(self):
        """
        Evalúa el modelo entrenado
        """
        if self.model is None or self.X_test is None:
            print("❌ Error: Modelo no entrenado o datos no disponibles")
            return
        
        print("📊 Evaluando modelo...")
        
        # Evaluación básica
        eval_results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        test_loss = eval_results[0]
        test_acc = eval_results[1]
        test_top3 = eval_results[2] if len(eval_results) > 2 else 0.0
        
        print(f"\n📈 Métricas de evaluación:")
        print(f"   🎯 Precisión: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"   🏆 Top-3 Precisión: {test_top3:.4f} ({test_top3*100:.2f}%)")
        print(f"   📉 Pérdida: {test_loss:.4f}")
        
        # Predicciones para matriz de confusión
        y_pred = self.model.predict(self.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Reporte de clasificación
        print(f"\n📋 Reporte de clasificación:")
        report = classification_report(
            y_true_classes, y_pred_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        for class_name in self.class_names:
            metrics = report[class_name]
            print(f"   {class_name}:")
            print(f"      Precisión: {metrics['precision']:.3f}")
            print(f"      Recall: {metrics['recall']:.3f}")
            print(f"      F1-score: {metrics['f1-score']:.3f}")
        
        return report
    
    def plot_training_history(self):
        """
        Grafica el historial de entrenamiento
        """
        if self.history is None:
            print("❌ No hay historial de entrenamiento disponible")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfica de precisión
        ax1.plot(self.history.history['accuracy'], label='Entrenamiento', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validación', linewidth=2)
        ax1.set_title('Precisión del Modelo', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Precisión')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfica de pérdida
        ax2.plot(self.history.history['loss'], label='Entrenamiento', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validación', linewidth=2)
        ax2.set_title('Pérdida del Modelo', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Pérdida')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('historial_entrenamiento.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráfica guardada como: historial_entrenamiento.png")
    
    def predict_single_image(self, image_path):
        """
        Predice la fruta en una imagen individual
        """
        if self.model is None:
            print("❌ Error: Modelo no disponible")
            return None
        
        try:
            # Cargar y procesar imagen
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Error: No se pudo cargar la imagen {image_path}")
                return None
            
            # Preprocesar
            img_resized = cv2.resize(img, (128, 128))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Predecir
            predictions = self.model.predict(img_batch, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            
            # Obtener información de precio
            price_info = self.fruit_prices.get(predicted_class, {
                'price': 0.00, 'unit': 'no disponible', 'currency': 'S/.'
            })
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'price': price_info,
                'all_predictions': predictions[0]
            }
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None
    
    def show_prediction_with_image(self, image_path):
        """
        Muestra la imagen con la predicción y precio
        """
        result = self.predict_single_image(image_path)
        
        if result is None:
            return
        
        # Cargar imagen original
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crear visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mostrar imagen
        ax1.imshow(img_rgb)
        ax1.set_title(f"Imagen Original", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Mostrar top 5 predicciones
        top_5_idx = np.argsort(result['all_predictions'])[-5:][::-1]
        top_5_classes = [self.class_names[i] for i in top_5_idx]
        top_5_probs = result['all_predictions'][top_5_idx]
        
        colors = ['green' if i == 0 else 'blue' for i in range(5)]
        bars = ax2.barh(range(5), top_5_probs, color=colors, alpha=0.7)
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(top_5_classes)
        ax2.set_xlabel('Probabilidad')
        ax2.set_title('Top 5 Predicciones', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, prob in zip(bars, top_5_probs):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        # Información en consola
        print(f"\n🍎 RESULTADO DE DETECCIÓN:")
        print(f"   Fruta: {result['class']}")
        print(f"   Confianza: {result['confidence']:.2%}")
        print(f"   💰 Precio: {result['price']['currency']} {result['price']['price']:.2f} por {result['price']['unit']}")
        
        plt.show()

def main():
    """
    Función principal del programa
    """
    print("🍎 DETECTOR DE FRUTAS CON PRECIOS")
    print("=" * 60)
    print("Proyecto de Percepción Computacional")
    print("Entrenamiento de CNN desde cero")
    print("=" * 60)
    
    # Crear detector
    detector = FruitPriceDetector()
    
    # Ruta del dataset
    dataset_path = r"d:\Carrera - Ing. Sistemas\Ciclo VI\percepcion\Dataset_Filtrado"
    
    print(f"\n📂 Dataset ubicado en: {dataset_path}")
    
    # Menú principal
    while True:
        print(f"\n📋 MENÚ PRINCIPAL:")
        print(f"1. 📊 Cargar y preparar datos")
        print(f"2. 🧠 Crear modelo CNN")
        print(f"3. 🏋️ Entrenar modelo")
        print(f"4. 📈 Evaluar modelo")
        print(f"5. 📊 Ver gráficas de entrenamiento")
        print(f"6. 🖼️ Probar con imagen individual")
        print(f"7. 💰 Ver precios de frutas")
        print(f"8. ❌ Salir")
        
        opcion = input(f"\n👉 Selecciona una opción (1-8): ").strip()
        
        if opcion == "1":
            print(f"\n🔄 Cargando datos...")
            success = detector.load_and_preprocess_data(dataset_path)
            if success:
                print(f"✅ Datos cargados correctamente")
            else:
                print(f"❌ Error al cargar datos")
        
        elif opcion == "2":
            if detector.class_names:
                detector.create_cnn_model()
            else:
                print(f"❌ Primero debes cargar los datos (opción 1)")
        
        elif opcion == "3":
            if detector.model is None:
                print(f"❌ Primero debes crear el modelo (opción 2)")
                continue
            
            epochs = input(f"📊 Número de épocas (recomendado: 30): ").strip()
            epochs = int(epochs) if epochs.isdigit() else 30
            
            batch_size = input(f"📦 Tamaño de lote (recomendado: 32): ").strip()
            batch_size = int(batch_size) if batch_size.isdigit() else 32
            
            detector.train_model(epochs=epochs, batch_size=batch_size)
        
        elif opcion == "4":
            detector.evaluate_model()
        
        elif opcion == "5":
            detector.plot_training_history()
        
        elif opcion == "6":
            if detector.model is None:
                print(f"❌ Primero debes entrenar el modelo")
                continue
            
            print(f"📂 Ingresa la ruta de la imagen:")
            image_path = input(f"👉 Ruta: ").strip().replace('"', '')
            
            if os.path.exists(image_path):
                detector.show_prediction_with_image(image_path)
            else:
                print(f"❌ Archivo no encontrado: {image_path}")
        
        elif opcion == "7":
            print(f"\n💰 PRECIOS ACTUALES DE FRUTAS:")
            print(f"-" * 50)
            for fruit, info in detector.fruit_prices.items():
                print(f"{fruit:.<30} {info['currency']} {info['price']:.2f} por {info['unit']}")
        
        elif opcion == "8":
            print(f"\n👋 ¡Gracias por usar el Detector de Frutas!")
            print(f"Proyecto desarrollado desde cero 🍎")
            break
        
        else:
            print(f"❌ Opción no válida. Selecciona 1-8.")

if __name__ == "__main__":
    main()