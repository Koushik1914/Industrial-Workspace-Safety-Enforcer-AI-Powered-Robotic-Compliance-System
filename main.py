import cv2
import numpy as np
import os
from ultralytics import YOLO
import RPi.GPIO as GPIO
from time import sleep
from RPLCD.gpio import CharLCD
from picamera2 import Picamera2

# =================== YOLO Model Setup ===================
MODEL_PATH = "/home/jayanth/yolov8s_custom.pt"  # Use "yolov8n.pt" for faster inference
model = YOLO(MODEL_PATH)

# =================== LCD Setup ===================
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

LCD_RS = 26
LCD_E = 19
LCD_DATA_PINS = [13, 6, 5, 11]
CONTRAST_PIN = 18  # LCD Contrast Control

# Setup LCD contrast control
GPIO.setup(CONTRAST_PIN, GPIO.OUT)
pwm = GPIO.PWM(CONTRAST_PIN, 1000)
pwm.start(50)  # Adjust contrast (40-70) if needed

lcd = CharLCD(
    numbering_mode=GPIO.BCM,  
    cols=16, 
    rows=2, 
    pin_rs=LCD_RS, 
    pin_e=LCD_E, 
    pins_data=LCD_DATA_PINS,
    auto_linebreaks=True  
)
lcd.clear()
lcd.write_string("PPE Detection...")

# =================== PPE Detection Requirements ===================
REQUIRED_PPE = {"Helmet", "Safety-Boot", "Safety-Vest"}  
CONFIDENCE_THRESHOLD = 0.5  

# Store last LCD message to prevent flickering
last_status = ""

def update_lcd(status):
    """Update LCD only if the message changes to prevent flickering."""
    global last_status
    if status != last_status:
        lcd.clear()
        lcd.write_string(status)
        last_status = status

# =================== Camera Setup ===================
print("Initializing Picamera2...")
picam2 = Picamera2()
camera_config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (640, 480)},  
    controls={"FrameRate": 30}
)
picam2.configure(camera_config)
picam2.start()
print("Camera initialized!")

# =================== Detection Loop ===================
try:
    while True:
        # Capture frame from Picamera2
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  

        # YOLO Detection
        results = model(frame)
        detected_ppe = set()  

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])  
                label = model.names.get(cls_id, "Unknown")  
                confidence = float(box.conf[0])

                if label in REQUIRED_PPE and confidence > CONFIDENCE_THRESHOLD:
                    detected_ppe.add(label)  

                # Draw detection boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # =================== LCD Message Logic ===================
        if detected_ppe == REQUIRED_PPE:
            update_lcd("Access Granted")
        else:
            update_lcd("Access Denied")

        # Display output on screen
        cv2.imshow("PPE Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

except Exception as e:
    print(f"Error: {e}")

finally:
    print("Cleaning up...")
    lcd.clear()
    pwm.stop()
    GPIO.cleanup()
    picam2.close()
    cv2.destroyAllWindows()
