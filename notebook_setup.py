# Configura sys.path e cwd para notebooks em research/.

import os
import sys
from pathlib import Path


def configurar():
    raiz = Path(__file__).resolve().parent
    src = raiz / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(raiz)
    return raiz
