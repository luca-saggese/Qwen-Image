import subprocess
import sys
from packaging import version

def should_install(req):
    """Verifica se un pacchetto va realmente installato"""
    if '==' not in req:
        return True
    
    pkg, ver = req.split('==')
    result = subprocess.run([sys.executable, '-m', 'pip', 'show', pkg], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        return True  # Non installato
        
    # Estrai versione installata
    for line in result.stdout.split('\n'):
        if line.startswith('Version:'):
            installed_ver = line.split()[1]
            # Confronta versioni
            if version.parse(installed_ver) >= version.parse(ver):
                print(f"✓ {pkg}>={ver} già soddisfatto ({installed_ver})")
                return False
    return True

with open('requirements.txt') as f:
    missing = [req.strip() for req in f if req.strip() and should_install(req.strip())]
    
if missing:
    # subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
    for package in missing:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', package])
