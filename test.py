import serial
import time
import serial.tools.list_ports

def find_arduino_port():
    # This function lists all connected USB devices
    ports = list(serial.tools.list_ports.comports())
    
    print("\n--- Scanning for Ports ---")
    if not ports:
        print("[ERROR] No USB devices found! Check your cable.")
        return None
    
    for p in ports:
        print(f"Found Device: {p.device} - {p.description}")
        
    # Ask user to pick one if multiple are found, or confirm the single one
    target_port = input("\nEnter the COM port to test (e.g., COM3): ").strip().upper()
    return target_port

def test_connection():
    port = find_arduino_port()
    if not port:
        return

    print(f"\n[INFO] Attempting to connect to {port} at 115200 baud...")
    
    try:
        # Establish connection
        arduino = serial.Serial(port, 115200, timeout=1)
        time.sleep(3) # WAIT for the board to reset (Crucial for ESP32/Arduino)
        print("[SUCCESS] Connected! Starting Signal Test...\n")

        print("Test 1: Sending 'S' (DANGER Signal)")
        print("--> Watch your Hardware: Buzzer SHOULD Beep, Screen 'DONT SLEEP'")
        arduino.write(b'S')
        
        time.sleep(5) # Wait 5 seconds to let you see/hear it
        
        print("\nTest 2: Sending 'O' (SAFE Signal)")
        print("--> Watch your Hardware: Buzzer SHOULD Stop, Screen 'VEHICLE READY'")
        arduino.write(b'O')
        
        time.sleep(2)
        
        print("\n[DONE] Test Finished. Closing connection.")
        arduino.close()

    except serial.SerialException as e:
        print(f"\n[CRITICAL ERROR] Could not open {port}.")
        print("Possible causes:")
        print("1. The Arduino IDE Serial Monitor is OPEN. Close it!")
        print("2. You selected the wrong COM port.")
        print(f"Error details: {e}")

if __name__ == "__main__":
    test_connection()