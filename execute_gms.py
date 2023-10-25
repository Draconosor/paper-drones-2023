import subprocess
import pyautogui
import time

# Path to the GAMSIDE executable (gamside.exe)
gamside_executable = r"C:\Users\Carlo\OneDrive\Documentos\23.9\gamside.exe"

# Path to your GAMS file (.gms)
gams_file = r'C:\Users\Carlo\OneDrive - Universidad de la Sabana\MGOP\DRONES 2023 2\paper-drones-2023\instances\C50P10T5D10\GENERICO NEOS.gms'
# Path to your GAMS Project file (.gpr)
gpr_file = r'C:\Users\Carlo\OneDrive - Universidad de la Sabana\MGOP\DRONES 2023 2\paper-drones-2023\instances\C50P10T5D10\C50P10T5D10.gpr'

subprocess.Popen([gamside_executable,gpr_file ,gams_file])

# Wait for GAMS IDE to open (you can adjust the delay as needed)
time.sleep(3)

# Simulate pressing the F5 key
pyautogui.press('F9')