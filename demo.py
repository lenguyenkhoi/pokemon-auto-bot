import sys
import subprocess
import os

def run_demo():
    print("=== CHAY DEMO POKEMON AUTO BOT - AI CHOI TRUC TIEP TREN GAME THAT ===")
    
    script_path = os.path.join("Pikachu-Matching-Game", "pikachu.py")
    if not os.path.exists(script_path):
        print(f"Loi: Khong tim thay file {script_path}")
        return
        
    print(f"Dang khoi dong game va bat che do AI tu dong choi...")
    # Chay game voi flag --ai
    cmd = [sys.executable, script_path, "--ai"]
    subprocess.run(cmd)

if __name__ == "__main__":
    run_demo()
