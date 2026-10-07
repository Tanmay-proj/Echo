import serial
import sys

port = "COM8"
baud = 115200

print(f"Opening {port} at {baud}...")
try:
    s = serial.Serial(port, baud, timeout=1.5)
except Exception as e:
    print(f"Failed to open {port}: {e}")
    sys.exit(1)

print("Reading 15 lines from board...")
for i in range(15):
    line = s.readline()
    print(f"[{i+1}] {repr(line)}")

s.close()
print("Done.")
