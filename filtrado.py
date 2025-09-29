# -*- coding: utf-8 -*-
"""
Filtrado de dataset Fruits-360 - Adaptado para entorno local Windows
Adaptado desde Google Colab para funcionar en PC local
"""

import os
import json
import zipfile
import shutil
import subprocess
import sys

def setup_kaggle():
    """Configura Kaggle API en Windows"""
    print("=== Configuración de Kaggle ===")
    
    # Crear directorio .kaggle en el home del usuario
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json_path):
        print(f"❌ No se encontró kaggle.json en: {kaggle_json_path}")
        print("Por favor:")
        print("1. Ve a Kaggle.com > Account > Create New API Token")
        print("2. Descarga kaggle.json")
        print(f"3. Colócalo en: {kaggle_json_path}")
        return False
    
    print(f"✅ kaggle.json encontrado en: {kaggle_json_path}")
    return True

def install_kaggle():
    """Instala kaggle si no está instalado"""
    try:
        import kaggle
        print("✅ Kaggle ya está instalado")
        return True
    except ImportError:
        print("Instalando kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("✅ Kaggle instalado")
        return True

def download_dataset():
    """Descarga el dataset de frutas desde Kaggle"""
    if not setup_kaggle():
        return None
    
    if not install_kaggle():
        return None
    
    # Directorio de descarga (en el directorio actual)
    download_dir = os.path.join(os.getcwd(), "fruits_dataset")
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Descargando dataset en: {download_dir}")
    
    try:
        # Ejecutar comando kaggle
        cmd = ["kaggle", "datasets", "download", "-d", "moltean/fruits", "-p", download_dir, "--unzip"]
        subprocess.run(cmd, check=True)
        print("✅ Dataset descargado y descomprimido")
        return download_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al descargar dataset: {e}")
        return None

# Intentar descargar el dataset
dataset_base = download_dataset()

# Buscar la carpeta Training en el dataset descargado
def find_training_folder(base_dir):
    """Encuentra la carpeta Training en el dataset descargado"""
    if not base_dir:
        print("❌ No se pudo descargar el dataset")
        return None
    
    # Posibles ubicaciones del Training después de descomprimir
    candidates = [
        os.path.join(base_dir, "fruits-360", "Training"),
        os.path.join(base_dir, "fruits-360", "fruits-360", "Training"),
        os.path.join(base_dir, "fruits-360_100x100", "fruits-360", "Training"),
        os.path.join(base_dir, "Training")  # En caso de que esté directamente
    ]
    
    for candidate in candidates:
        if os.path.isdir(candidate):
            print(f"📂 Training localizado en: {candidate}")
            print("Ejemplo de clases:", os.listdir(candidate)[:10])
            return candidate
    
    # Si no se encuentra, mostrar qué hay en el directorio base
    print(f"❌ No se encontró la carpeta 'Training' en: {base_dir}")
    if os.path.exists(base_dir):
        print("Contenido disponible:", os.listdir(base_dir))
    return None

dataset_origen = find_training_folder(dataset_base)
if not dataset_origen:
    print("❌ No se puede continuar sin la carpeta Training")
    exit(1)

import os, shutil, re

# 1) Muestra cómo se llaman realmente tus clases
print("Ejemplo de clases en Training:")
print(sorted(os.listdir(dataset_origen))[:60])  # dataset_origen ya lo tenías

# 2) Palabras clave a buscar (en minúsculas) -> sinónimos
wish = {
    "apple red 1": ["apple red 1","apple 1","apple 10","apple 11","apple red"],
    "banana": ["banana"],
    "melon 2": ["cantaloupe 2","cantaloupe","melon","piel de sapo","galia"],
    "cocos": ["cocos","coconut"],
    "granadilla": ["granadilla"],
    "kiwi": ["kiwi"],
    "passion fruit": ["passion fruit","maracuya","maracuyá"],
    "orange": ["orange"],
    "pear forelle": ["pear forelle","forelle","pear  "],
    "pineapple": ["pineapple","piña","pina"]
}

# 3) Normalizador
def norm(s): return re.sub(r"\s+"," ",s.strip().lower())

# 4) Índice de carpetas reales
classes_real = os.listdir(dataset_origen)
classes_norm = {norm(c): c for c in classes_real}  # norm -> original

# 5) Buscar mejor match por “contiene”
def find_match(keywords):
    kws = [norm(k) for k in keywords]
    # prioridad: coincidencia completa; si no, contiene
    for k in kws:
        if k in classes_norm:
            return classes_norm[k]
    for k in kws:
        for cn, original in classes_norm.items():
            if k in cn:
                return original
    return None

# Configurar carpeta destino (ajustar según tu estructura)
destino_base = r"d:\Carrera - Ing. Sistemas\Ciclo VI\percepcion"
dataset_destino = os.path.join(destino_base, "Dataset_Filtrado")
os.makedirs(dataset_destino, exist_ok=True)
print(f"📁 Dataset filtrado se guardará en: {dataset_destino}")

copiadas, faltantes = [], []

for esp, keywords in wish.items():
    found = find_match(keywords)
    if found and os.path.isdir(os.path.join(dataset_origen, found)):
        src = os.path.join(dataset_origen, found)
        dst = os.path.join(dataset_destino, found)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        copiadas.append((esp, found))
        print(f"✅ Copiada: '{esp}' -> '{found}'")
    else:
        faltantes.append((esp, keywords))
        print(f"⚠️ No se encontró match para: {esp} | probé: {keywords}")

print("\n" + "="*50)
print("RESUMEN FINAL")
print("="*50)
print(f"✅ Clases copiadas ({len(copiadas)}):", [c[1] for c in copiadas])
if faltantes:
    print(f"⚠️ No encontradas ({len(faltantes)}):", [f[0] for f in faltantes])
print(f"📂 Dataset filtrado guardado en: {dataset_destino}")
print("="*50)

def main():
    """Función principal para ejecutar todo el proceso"""
    print("🍎 Iniciando filtrado de dataset Fruits-360...")
    
    # El proceso ya se ejecutó arriba, esta función es para referencia
    if dataset_origen and os.path.exists(dataset_destino):
        total_classes = len(os.listdir(dataset_destino))
        print(f"✅ Proceso completado. {total_classes} clases filtradas.")
    else:
        print("❌ Proceso no completado correctamente.")

if __name__ == "__main__":
    main()