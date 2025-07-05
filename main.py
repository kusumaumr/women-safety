import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import pyttsx3
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
import threading
import geocoder
import requests
import os
import json
import webbrowser

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Improved gender detection model
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderList = ['Male', 'Female']
genderNet = cv2.dnn.readNet(genderModel, genderProto)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Face alignment parameters
FACE_IMG_SIZE = 128
LEFT_EYE_INDEX = 33  # MediaPipe left eye landmark index
RIGHT_EYE_INDEX = 263  # MediaPipe right eye landmark index

# Alert settings
alert_cooldown = 5  # seconds between voice alerts-
last_voice_alert_time = 0
alarm_triggered = False
last_alert_time = 0
alert_cooldown_period = 30  # seconds between full alerts

# Twilio and Gmail credentials (replace with your actual credentials)
TWILIO_ACCOUNT_SID = 'AC6178cb6629a2f51843634e3ff7439fb4'
TWILIO_AUTH_TOKEN = '5f502b8f1886494a76af05497c241ee0'
TWILIO_PHONE_NUMBER = '+18154860843'
RECIPIENT_PHONE = '+917842501571'
GMAIL_USER = 'kusumaumr@gmail.com'
GMAIL_PASSWORD = 'bogt frfs dpin wtva'
RECIPIENT_EMAIL = 'tamminenimadhavi05@gmail.com'

# Initialize Twilio client
try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
except Exception as e:
    print(f"Failed to initialize Twilio client: {e}")
    twilio_client = None

# Initialize text-to-speech
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Google Maps API (replace with your API key)
GOOGLE_MAPS_API_KEY = 'YOUR_ACTUAL_KEY_HERE'
GEOCODING_API_URL = 'https://maps.googleapis.com/maps/api/geocode/json'

# Location variables
current_location = "Initializing location..."
current_gmaps_link = "https://maps.google.com"
last_location_update = 0
location_update_interval = 300  # 5 minutes

def align_face(face_roi, landmarks):
    """Align face using eye landmarks for better gender detection"""
    try:
        # Get eye landmarks
        left_eye = landmarks[LEFT_EYE_INDEX]
        right_eye = landmarks[RIGHT_EYE_INDEX]
        
        # Calculate angle between eyes
        dY = right_eye.y - left_eye.y
        dX = right_eye.x - left_eye.x
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        # Get center between eyes
        eyes_center = ((left_eye.x + right_eye.x) // 2, (left_eye.y + right_eye.y) // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        
        # Apply affine transformation
        aligned_face = cv2.warpAffine(face_roi, M, (face_roi.shape[1], face_roi.shape[0]), 
                                     flags=cv2.INTER_CUBIC)
        
        return aligned_face
    except Exception as e:
        print(f"Face alignment error: {e}")
        return face_roi

def get_google_maps_location(lat, lng):
    """Get precise address using Google Maps API"""
    try:
        params = {
            'latlng': f"{lat},{lng}",
            'key': GOOGLE_MAPS_API_KEY
        }
        response = requests.get(GEOCODING_API_URL, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK':
                return {
                    'address': data['results'][0]['formatted_address'],
                    'maps_link': f"https://www.google.com/maps/?q={lat},{lng}"
                }
        return {
            'address': f"Near {lat:.4f}, {lng:.4f}",
            'maps_link': f"https://www.google.com/maps/?q={lat},{lng}"
        }
    except Exception as e:
        print(f"Google Maps API error: {e}")
        return {
            'address': "Location service unavailable",
            'maps_link': "https://maps.google.com"
        }

def get_current_location():
    """Get current location with multiple fallback methods"""
    try:
        # First try with Google Maps API
        g = geocoder.ip('me')
        if g.ok:
            lat, lng = g.latlng
            location_data = get_google_maps_location(lat, lng)
            return location_data['address'], location_data['maps_link']
        
        # Fallback to basic geocoding
        return f"Near {lat:.4f}, {lng:.4f}", f"https://www.google.com/maps/?q={lat},{lng}"
    except Exception as e:
        print(f"Location error: {e}")
        return "Location service unavailable", "https://maps.google.com"

def update_location():
    """Update the current location and Google Maps link"""
    global current_location, current_gmaps_link, last_location_update
    current_location, current_gmaps_link = get_current_location()
    last_location_update = time.time()
    print(f"Location updated: {current_location}")

def speak(message):
    """Function to speak the given message"""
    try:
        tts_engine.say(message)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def send_sms(message):
    """Function to send SMS alert"""
    if twilio_client is None:
        print("Twilio client not initialized - SMS not sent")
        return
        
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE
        )
        print("SMS sent successfully")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

def send_email(subject, body):
    """Enhanced email with Google Maps link"""
    msg = MIMEText(body + f"\n\nGoogle Maps: {current_gmaps_link}")
    msg['Subject'] = subject
    msg['From'] = GMAIL_USER
    msg['To'] = RECIPIENT_EMAIL
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, RECIPIENT_EMAIL, msg.as_string())
        print("Email with map link sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def trigger_alert(message, alert_type="warning"):
    """Enhanced alerts with precise location"""
    global last_voice_alert_time, alarm_triggered, last_alert_time
    
    # Update location if needed
    if time.time() - last_location_update > location_update_interval:
        update_location()
    
    # Add location to message
    full_message = f"{message} at {current_location}"
    
    # Voice alert
    if time.time() - last_voice_alert_time > alert_cooldown:
        threading.Thread(target=speak, args=(full_message,)).start()
        last_voice_alert_time = time.time()
    
    # Enhanced SMS with map link (only if enough time has passed since last alert)
    if alert_type == "critical" and not alarm_triggered and time.time() - last_alert_time > alert_cooldown_period:
        alarm_triggered = True
        last_alert_time = time.time()
        sms_message = f"{full_message}. Map: {current_gmaps_link}"
        threading.Thread(target=send_sms, args=(sms_message,)).start()
        
        email_subject = f"URGENT: {message}"
        email_body = f"""Alert Details:
        - Message: {message}
        - Location: {current_location}
        - Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - Map: {current_gmaps_link}
        """
        threading.Thread(target=send_email, args=(email_subject, email_body)).start()

def isPalmFacingCamera(hand_landmarks):
    """Improved palm facing detection using multiple landmarks"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Check if fingertips are above the wrist (palm facing camera)
    return (index_tip.y < wrist.y and 
            middle_tip.y < wrist.y and 
            ring_tip.y < wrist.y and
            pinky_tip.y < wrist.y)

def preprocess_face(face_roi):
    """Preprocess face for gender detection"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram
        gray = cv2.equalizeHist(gray)
        
        # Resize to model input size
        resized = cv2.resize(gray, (227, 227))
        
        # Convert to 3 channels (required by the model)
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        
        return resized
    except Exception as e:
        print(f"Face preprocessing error: {e}")
        return face_roi

def detect_gender(face_roi, landmarks=None):
    """Improved gender detection with face alignment and preprocessing"""
    if face_roi.size == 0 or face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
        return "Unknown", 0.0
    
    try:
        # Align face if landmarks are available
        if landmarks is not None:
            face_roi = align_face(face_roi, landmarks)
        
        # Preprocess face
        processed_face = preprocess_face(face_roi)
        
        # Create blob and run through network
        blob = cv2.dnn.blobFromImage(processed_face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        
        # Get gender and confidence
        gender = genderList[genderPreds[0].argmax()]
        confidence = genderPreds[0].max()
        
        # Only return if confidence is high enough
        if confidence > 0.7:
            return gender, confidence
        else:
            return "Unknown", confidence
    except Exception as e:
        print(f"Gender detection error: {e}")
        return "Unknown", 0.0

# Get initial location
update_location()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables for hand raise detection
raise_hand_start_time = 0
raise_hand_duration_threshold = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    
    # Reset counters
    male_count = 0
    female_count = 0
    raise_hand_detected = False
    face_boxes = []
    genders = []
    status = "Monitoring..."
    status_color = (255, 255, 255)
    proximity_warning = False

    # Hand detection
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if isPalmFacingCamera(hand_landmarks):
                if raise_hand_start_time == 0:
                    raise_hand_start_time = time.time()
                elif time.time() - raise_hand_start_time > raise_hand_duration_threshold:
                    raise_hand_detected = True
                    
                    # Visual feedback
                    cx, cy = 0, 0
                    for landmark in hand_landmarks.landmark:
                        cx += landmark.x
                        cy += landmark.y
                    cx = int(cx * w / len(hand_landmarks.landmark))
                    cy = int(cy * h / len(hand_landmarks.landmark))
                    
                    cv2.circle(frame, (cx, cy), 30, (0, 0, 255), -1)
                    cv2.putText(frame, "HELP!", (cx-20, cy+10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                raise_hand_start_time = 0

    # Face detection and gender classification
    face_results = face_detection.process(rgb_frame)
    face_mesh_results = face_mesh.process(rgb_frame)
    
    if face_results.detections:
        for i, detection in enumerate(face_results.detections):
            bboxC = detection.location_data.relative_bounding_box
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = int((bboxC.xmin + bboxC.width) * w)
            y2 = int((bboxC.ymin + bboxC.height) * h)
            
            # Ensure coordinates are within frame bounds and valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
                
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
                
            # Get face landmarks for alignment if available
            landmarks = None
            if face_mesh_results.multi_face_landmarks and i < len(face_mesh_results.multi_face_landmarks):
                landmarks = face_mesh_results.multi_face_landmarks[i].landmark
                
            # Detect gender with improved method
            gender, confidence = detect_gender(face_roi, landmarks)
            
            face_boxes.append((x1, y1, x2, y2))
            genders.append(gender)
            
            # Draw face info
            color = (255, 0, 0) if gender == 'Male' else (0, 0, 255) if gender == 'Female' else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if gender != "Unknown":
                label = f"{gender} ({confidence:.1%})"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Update counts
            if gender == 'Male':
                male_count += 1
            elif gender == 'Female':
                female_count += 1

    # Check proximity between males and females
    for i, (box1, gender1) in enumerate(zip(face_boxes, genders)):
        for j, (box2, gender2) in enumerate(zip(face_boxes, genders)):
            if i != j and gender1 == 'Female' and gender2 == 'Male':
                # Simple proximity check (center distance)
                center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
                center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
                distance = np.sqrt((center1[0] - center2[0])*2 + (center1[1] - center2[1])*2)
                
                if distance < 0.2 * w:  # 20% of frame width
                    proximity_warning = True
                    cv2.putText(frame, "Warning: Male near Female!", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Determine status and trigger alerts
    if raise_hand_detected:
        status = "HARASSMENT ALERT! Raise hand detected"
        status_color = (0, 0, 255)
        
        # Flash red background during alert
        if int(time.time() * 2) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        cv2.putText(frame, "HARASSMENT ALERT!", (w//2-200, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        
        trigger_alert("EMERGENCY! Someone raised their hand for help!", "critical")
    elif proximity_warning:
        status = "Warning: Male close to Female! Be careful"
        status_color = (0, 165, 255)
        trigger_alert("Warning! A male is too close to a female. Please be careful.")
    elif female_count > 0 and male_count > 0:
        status = "Multiple people detected. Be safe"
        status_color = (0, 255, 255)
        if time.time() - last_voice_alert_time > alert_cooldown * 2:
            trigger_alert("Multiple people detected. Please stay safe.")
    elif female_count > 0:
        status = "Female detected. All safe"
        status_color = (0, 255, 0)
        if time.time() - last_voice_alert_time > alert_cooldown * 2:
            trigger_alert("Female detected. You are safe now.")
    elif male_count > 0:
        status = "Male detected. Monitoring..."
        status_color = (255, 255, 0)
        if time.time() - last_voice_alert_time > alert_cooldown * 2:
            trigger_alert("Male detected. Monitoring the situation.")
    else:
        status = "No one detected"
        status_color = (255, 255, 255)
        alarm_triggered = False
        raise_hand_start_time = 0  # Reset hand raise timer when no one is detected

    # Display status with clickable map link in console
    print(f"Current location: {current_location}")
    print(f"Google Maps link: {current_gmaps_link}")
    
    # Display on frame
    cv2.putText(frame, status, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"Males: {male_count} | Females: {female_count}", (10, h-50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Location: {current_location[:40]}...", (10, h-80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Press 'M' to view map", (w-200, h-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

    # Show frame
    cv2.imshow('Women Safety Monitoring', frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('m'):
        print(f"Opening map: {current_gmaps_link}")
        webbrowser.open(current_gmaps_link)

    # Update location periodically
    if time.time() - last_location_update > location_update_interval:
        threading.Thread(target=update_location).start()

cap.release()
cv2.destroyAllWindows()