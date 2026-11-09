from src.part_a import run as part_a
from src.part_b import run as part_b
from src.part_c import run as part_c
from src.part_d import run as part_d
import subprocess, sys
if __name__=='__main__':
    part_a(); subprocess.run([sys.executable, "src/validate.py"], check=True); part_b(); part_c(); part_d()
