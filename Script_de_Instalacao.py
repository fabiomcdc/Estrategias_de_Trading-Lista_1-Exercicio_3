# instalando as dependencies

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Lista de pacotes a serem instalados
packages = ["pandas", "scipy", "matplotlib.pyplot"]

for package in packages:
    install(package)

print("Dependências instaladas com sucesso!")