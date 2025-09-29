# -*- coding: utf-8 -*-
"""
Script simplificado para descargar y filtrar dataset Fruits-360
Versión con mejor manejo de errores y timeouts
"""

import os
import subprocess
import sys
import time

def test_kaggle_connection():
    """Prueba la conexión a Kaggle con un comando simple"""
    print("🔍 Probando conexión a Kaggle...")
    try:
        result = subprocess.run(["kaggle", "--version"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Kaggle CLI funciona correctamente")
            return True
        else:
            print(f"❌ Error en Kaggle CLI: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error al probar Kaggle: {e}")
        return False

def download_fruits_dataset():
    """Descarga el dataset con manejo de timeouts mejorado"""
    if not test_kaggle_connection():
        return False
    
    download_dir = os.path.join(os.getcwd(), "fruits_dataset")
    print(f"📁 Directorio de descarga: {download_dir}")
    
    # Limpiar directorio anterior
    if os.path.exists(download_dir):
        import shutil
        shutil.rmtree(download_dir)
        print("🧽 Directorio anterior eliminado")
    
    os.makedirs(download_dir, exist_ok=True)
    
    print("⬇️ Iniciando descarga del dataset Fruits-360...")
    print("ℹ️ Esto puede tardar varios minutos (el dataset es grande ~1GB)")
    
    try:
        # Comando sin --unzip para descargar más rápido
        cmd = ["kaggle", "datasets", "download", "-d", "moltean/fruits", "-p", download_dir]
        print(f"🚀 Ejecutando: {' '.join(cmd)}")
        
        # Ejecutar con timeout más largo
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, text=True)
        
        # Esperar con indicador de progreso
        dots = 0
        while process.poll() is None:
            print(f"\r⏳ Descargando{'.' * (dots % 4)}{' ' * (3 - (dots % 4))}", end='', flush=True)
            time.sleep(2)
            dots += 1
            
            # Timeout de 10 minutos
            if dots > 300:  # 10 minutos * 30 (cada 2 segundos)
                process.terminate()
                print("\n❌ Timeout: Descarga cancelada después de 10 minutos")
                return False
        
        stdout, stderr = process.communicate()
        print()  # Nueva línea después del indicador de progreso
        
        if process.returncode == 0:
            print("✅ Descarga completada")
            
            # Verificar archivos descargados
            items = os.listdir(download_dir)
            if items:
                print(f"📦 Archivos descargados ({len(items)}):")
                for item in items:
                    item_path = os.path.join(download_dir, item)
                    if os.path.isfile(item_path):
                        size_mb = os.path.getsize(item_path) / (1024*1024)
                        print(f"  - {item} ({size_mb:.1f} MB)")
                return True
            else:
                print("❌ No se descargaron archivos")
                return False
        else:
            print(f"❌ Error en descarga:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def extract_dataset():
    """Extrae el dataset descargado"""
    download_dir = os.path.join(os.getcwd(), "fruits_dataset")
    
    # Buscar archivo zip
    zip_files = [f for f in os.listdir(download_dir) if f.endswith('.zip')]
    
    if not zip_files:
        print("❌ No se encontró archivo ZIP para extraer")
        return False
    
    zip_file = zip_files[0]
    zip_path = os.path.join(download_dir, zip_file)
    
    print(f"📦 Extrayendo {zip_file}...")
    
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        
        print("✅ Extracción completada")
        
        # Mostrar estructura
        print("📂 Estructura extraída:")
        for root, dirs, files in os.walk(download_dir):
            level = root.replace(download_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            if level < 2:  # Solo mostrar 2 niveles para no saturar
                subindent = ' ' * 2 * (level + 1)
                for f in files[:5]:  # Solo primeros 5 archivos
                    print(f"{subindent}{f}")
                if len(files) > 5:
                    print(f"{subindent}... y {len(files)-5} archivos más")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al extraer: {e}")
        return False

def main():
    """Función principal"""
    print("🍎 DESCARGADOR DE DATASET FRUITS-360")
    print("=" * 50)
    
    # Cambiar al directorio correcto
    target_dir = r"d:\Carrera - Ing. Sistemas\Ciclo VI\percepcion"
    if os.path.exists(target_dir):
        os.chdir(target_dir)
        print(f"📁 Directorio actual: {os.getcwd()}")
    else:
        print(f"❌ No se encontró el directorio: {target_dir}")
        return
    
    # Paso 1: Descargar
    if download_fruits_dataset():
        print("\n🎉 Descarga exitosa!")
        
        # Paso 2: Extraer
        if extract_dataset():
            print("\n🎉 Dataset listo para usar!")
            print("📋 Próximo paso: ejecutar filtrado.py para filtrar las clases deseadas")
        else:
            print("\n⚠️ Descarga OK, pero falló la extracción")
    else:
        print("\n❌ Falló la descarga")
        print("💡 Posibles soluciones:")
        print("  - Verificar conexión a internet")
        print("  - Verificar credenciales de Kaggle")
        print("  - Intentar más tarde (el servidor puede estar ocupado)")

if __name__ == "__main__":
    main()