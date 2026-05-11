import pygame

def test_audio_file(filename):
    print(f"Initializing Pygame mixer...")
    
    # Pre-init helps prevent audio stuttering/failure on Raspberry Pi
    pygame.mixer.pre_init(44100, -16, 2, 2048) 
    pygame.mixer.init()

    print(f"Loading '{filename}'...")
    try:
        pygame.mixer.music.load(filename)
        
        print("Playing audio... (Press Ctrl+C to stop)")
        pygame.mixer.music.play()
        
        # This loop keeps the Python script running as long as the audio is playing.
        # Without this, the script ends instantly and cuts off the sound!
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        print("Playback finished!")

    except pygame.error as e:
        print(f"\n[ERROR] Pygame could not load/play the file.")
        print(f"Details: {e}")
        print("Check if the file path is correct and the file isn't corrupted.")

if __name__ == "__main__":
    # Replace 'alert.wav' with your actual audio file's name
    # If the file is in a different folder, use the full path like:
    # '/home/iotlab/Desktop/DMS/Driver-Monitoring System/alert.wav'
    AUDIO_FILE = r"sound files/alarm.wav" 
    
    test_audio_file(AUDIO_FILE)
