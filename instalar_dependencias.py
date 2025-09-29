# -*- coding: utf-8 -*-
"""
Instalador automático de dependencias
Para el proyecto Detector de Frutas con Precios
"""

import subprocess
import sys
import os

def install_package(package):
    """Instala un paquete usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_installation():
    """Verifica que todas las dependencias estén instaladas"""
    packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'numpy': 'NumPy',
        'PIL': 'Pillow'
    }
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name} - Instalado")
        except ImportError:
            print(f"❌ {name} - Faltante")
            missing.append(package)
    
    return missing

def main():
    print("🍎 INSTALADOR DE DEPENDENCIAS")
    print("Detector de Frutas con Precios")
    print("=" * 40)
    
    print("\n🔍 Verificando dependencias actuales...")
    missing = check_installation()
    
    if not missing:
        print("\n🎉 ¡Todas las dependencias están instaladas!")
        return
    
    print(f"\n⚠️ Faltan {len(missing)} dependencias")
    install = input("¿Deseas instalarlas automáticamente? (s/n): ").lower()
    
    if install == 's' or install == 'si':
        print("\n📦 Instalando dependencias...")
        
        # Instalar desde requirements.txt si existe
        if os.path.exists("requirements.txt"):
            print("📋 Instalando desde requirements.txt...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                print("✅ Instalación completada desde requirements.txt")
            except subprocess.CalledProcessError:
                print("❌ Error instalando desde requirements.txt")
                print("🔄 Intentando instalación individual...")
                
                # Paquetes individuales
                packages_to_install = [
                    "tensorflow>=2.12.0",
                    "opencv-python>=4.8.0", 
                    "matplotlib>=3.7.0",
                    "seaborn>=0.12.0",
                    "scikit-learn>=1.3.0",
                    "numpy>=1.24.0",
                    "pillow>=10.0.0"
                ]
                
                for package in packages_to_install:
                    print(f"📦 Instalando {package}...")
                    if install_package(package):
                        print(f"✅ {package} instalado")
                    else:
                        print(f"❌ Error instalando {package}")
        else:
            print("❌ No se encontró requirements.txt")
        
        print("\n🔍 Verificando instalación final...")
        final_missing = check_installation()
        
        if not final_missing:
            print("\n🎉 ¡Instalación completada exitosamente!")
            print("🚀 Ya puedes ejecutar: python detector_frutas_desde_cero.py")
        else:
            print(f"\n⚠️ Aún faltan algunas dependencias: {final_missing}")
            print("💡 Intenta instalarlas manualmente con:")
            for pkg in final_missing:
                print(f"   pip install {pkg}")
    
    else:
        print("\n📋 Para instalar manualmente, ejecuta:")
        print("   pip install -r requirements.txt")
        print("O instala cada paquete individualmente:")
        print("   pip install tensorflow opencv-python matplotlib seaborn scikit-learn numpy pillow")

if __name__ == "__main__":
    main()