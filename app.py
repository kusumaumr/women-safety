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
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load your trained gender classification model
try:
    gender_model = load_model('gender_classification_model.h5')
    print("Successfully loaded custom gender classification model")
except Exception as e:
    print(f"Error loading custom model: {e}")
    # Fallback simple model if your model isn't found
    gender_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    gender_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

gender_labels = ['Male', 'Female']
FACE_IMG_SIZE = 128  # Should match your model's input size

# Face alignment parameters
LEFT_EYE_INDEX = 33  # MediaPipe left eye landmark index
RIGHT_EYE_INDEX = 263  # MediaPipe right eye landmark index

# Alert settings
alert_cooldown = 5  # seconds between voice alerts
last_voice_alert_time = 0
alarm_triggered = False
last_alert_time = 0
alert_cooldown_period = 30  # seconds between full alerts

# Twilio and Gmail credentials
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

# Location variables
current_location = "Initializing location..."
current_gmaps_link = "https://maps.google.com"
last_location_update = 0
location_update_interval = 300  # 5 minutes

# Face tracking
face_tracker = {}
face_id_counter = 0
FACE_TRACKING_DURATION = 5  # seconds to remember a face

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

def preprocess_face(face_roi):
    """Preprocess face for your gender classification model"""
    try:
        # Convert to RGB (if your model expects RGB)
        if len(face_roi.shape) == 2:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        elif face_roi.shape[2] == 4:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGRA2RGB)
        else:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(face_roi, (FACE_IMG_SIZE, FACE_IMG_SIZE))
        
        # Normalize pixel values
        normalized = resized.astype('float32') / 255.0
        
        # Expand dimensions if needed
        if len(normalized.shape) == 3:
            normalized = np.expand_dims(normalized, axis=0)
            
        return normalized
    except Exception as e:
        print(f"Face preprocessing error: {e}")
        return None

def detect_gender(face_roi, landmarks=None, face_id=None):
    """Gender detection using your custom model with face tracking"""
    if face_roi.size == 0 or face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
        return "Unknown", 0.0
    
    # Check if we have cached gender for this face
    if face_id is not None and face_id in face_tracker:
        if time.time() - face_tracker[face_id]['last_seen'] < FACE_TRACKING_DURATION:
            return face_tracker[face_id]['gender'], face_tracker[face_id]['confidence']
    
    try:
        # Align face if landmarks are available
        if landmarks is not None:
            face_roi = align_face(face_roi, landmarks)
        
        # Preprocess face
        processed_face = preprocess_face(face_roi)
        if processed_face is None:
            return "Unknown", 0.0
        
        # Make prediction
        predictions = gender_model.predict(processed_face)
        gender_idx = np.argmax(predictions)
        confidence = predictions[0][gender_idx]
        gender = gender_labels[gender_idx]
        
        # Only return if confidence is high enough
        if confidence > 0.7:
            # Cache the result if we have a face ID
            if face_id is not None:
                face_tracker[face_id] = {
                    'gender': gender,
                    'confidence': float(confidence),
                    'last_seen': time.time()
                }
            return gender, float(confidence)
        else:
            return "Unknown", float(confidence)
    except Exception as e:
        print(f"Gender detection error: {e}")
        return "Unknown", 0.0

def assign_face_id(face_center, face_size):
    """Track faces across frames"""
    global face_id_counter
    
    current_time = time.time()
    
    # Clean up old face entries
    global face_tracker
    face_tracker = {fid: data for fid, data in face_tracker.items() 
                   if current_time - data['last_seen'] < FACE_TRACKING_DURATION}
    
    # Find closest existing face
    for fid, data in face_tracker.items():
        dist = np.sqrt((data['center'][0] - face_center[0])**2 + 
                      (data['center'][1] - face_center[1])**2)
        if dist < 50 and abs(data['size'] - face_size) < 30:
            data['center'] = face_center
            data['size'] = face_size
            data['last_seen'] = current_time
            return fid
    
    # Create new face ID if no match found
    new_id = f"face_{face_id_counter}"
    face_id_counter += 1
    face_tracker[new_id] = {
        'center': face_center,
        'size': face_size,
        'last_seen': current_time,
        'gender': 'Unknown',
        'confidence': 0.0
    }
    return new_id

def get_current_location():
    """Get current location with geocoder"""
    try:
        g = geocoder.ip('me')
        if g.ok:
            lat, lng = g.latlng
            return f"Near {lat:.4f}, {lng:.4f}", f"https://www.google.com/maps/?q={lat},{lng}"
        return "Unavailable", "https://maps.google.com"
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
        print("SMS alert sent successfully")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

def send_email(subject, body):
    """Send email alert with location details"""
    msg = MIMEText(body + f"\n\nLocation: {current_location}\nGoogle Maps: {current_gmaps_link}")
    msg['Subject'] = subject
    msg['From'] = GMAIL_USER
    msg['To'] = RECIPIENT_EMAIL
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)
        print("Email alert sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def trigger_alert(message, alert_type="warning"):
    """Handle different types of alerts"""
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
    
    # Critical alerts (SMS + email)
    if alert_type == "critical" and not alarm_triggered and time.time() - last_alert_time > alert_cooldown_period:
        alarm_triggered = True
        last_alert_time = time.time()
        
        # Send SMS with map link
        sms_message = f"{full_message}. Map: {current_gmaps_link}"
        threading.Thread(target=send_sms, args=(sms_message,)).start()
        
        # Send detailed email
        email_subject = "URGENT: Safety Alert Triggered"
        email_body = f"""Alert Details:
        - Type: {message}
        - Location: {current_location}
        - Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - Map Link: {current_gmaps_link}
        """
        threading.Thread(target=send_email, args=(email_subject, email_body)).start()

def isPalmFacingCamera(hand_landmarks):
    """Check if palm is facing the camera (help signal)"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    fingers = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    ]
    return all(finger.y < wrist.y for finger in fingers)

def main():
    cap = cv2.VideoCapture(0)
    hand_raise_start_time = 0
    HAND_RAISE_DURATION = 2  # seconds
    
    # Harassment monitoring
    harassment_monitor = {
        'male_near_female': {},
        'harassment_alerts': set()
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Reset counters
        male_count = 0
        female_count = 0
        hand_raised = False
        harassment_detected = False
        
        # Process hands
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if isPalmFacingCamera(hand_landmarks):
                    if hand_raise_start_time == 0:
                        hand_raise_start_time = time.time()
                    elif time.time() - hand_raise_start_time > HAND_RAISE_DURATION:
                        hand_raised = True
                        # Visual feedback
                        cx, cy = int(w/2), int(h/2)
                        cv2.circle(frame, (cx, cy), 30, (0, 0, 255), -1)
                        cv2.putText(frame, "HELP!", (cx-30, cy+10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    hand_raise_start_time = 0
        
        # Process faces
        face_results = face_detection.process(rgb_frame)
        face_mesh_results = face_mesh.process(rgb_frame)
        
        face_data = []
        if face_results.detections:
            for i, detection in enumerate(face_results.detections):
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # Validate coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                if x1 >= x2 or y1 >= y2:
                    continue
                
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                
                # Assign face ID
                face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                face_size = (x2 - x1) * (y2 - y1)
                face_id = assign_face_id(face_center, face_size)
                
                # Get face landmarks if available
                landmarks = None
                if face_mesh_results.multi_face_landmarks and i < len(face_mesh_results.multi_face_landmarks):
                    landmarks = face_mesh_results.multi_face_landmarks[i].landmark
                
                # Detect gender using your custom model
                gender, confidence = detect_gender(face_img, landmarks, face_id)
                
                # Update counts
                if gender == 'Male':
                    male_count += 1
                    color = (255, 0, 0)  # Blue
                elif gender == 'Female':
                    female_count += 1
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 255, 255)  # Yellow (unknown)
                
                # Draw face box and info
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if gender != "Unknown":
                    label = f"{gender} ({confidence:.0%})"
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                face_data.append((face_id, gender, face_center))
        
        # Check for harassment situations
        current_time = time.time()
        harassment_monitor['male_near_female'] = {
            k: v for k, v in harassment_monitor['male_near_female'].items()
            if current_time - v < HARASSMENT_DURATION * 2
        }
        
        # Check proximity between males and females
        for i, (id1, gender1, center1) in enumerate(face_data):
            for j, (id2, gender2, center2) in enumerate(face_data):
                if i != j and ((gender1 == 'Male' and gender2 == 'Female') or 
                              (gender1 == 'Female' and gender2 == 'Male')):
                    
                    distance = np.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)
                    key = tuple(sorted((id1, id2)))
                    
                    if distance < PROXIMITY_THRESHOLD * w:
                        if key not in harassment_monitor['male_near_female']:
                            harassment_monitor['male_near_female'][key] = current_time
                        elif current_time - harassment_monitor['male_near_female'][key] > HARASSMENT_DURATION:
                            if key not in harassment_monitor['harassment_alerts']:
                                harassment_monitor['harassment_alerts'].add(key)
                                harassment_detected = True
                                cv2.putText(frame, "HARASSMENT ALERT!", (50, 50),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        if key in harassment_monitor['male_near_female']:
                            del harassment_monitor['male_near_female'][key]
                        if key in harassment_monitor['harassment_alerts']:
                            harassment_monitor['harassment_alerts'].remove(key)
        
        # Trigger alerts
        if hand_raised:
            trigger_alert("EMERGENCY! Help signal detected", "critical")
        elif harassment_detected:
            trigger_alert("WARNING! Male too close to female for extended period", "critical")
        
        # Display status
        status = f"Males: {male_count} | Females: {female_count}"
        if hand_raised:
            status = "EMERGENCY! Help signal detected"
            status_color = (0, 0, 255)
        elif harassment_detected:
            status = "HARASSMENT ALERT!"
            status_color = (0, 0, 255)
        elif female_count > 0 and male_count > 0:
            status = "Multiple people detected"
            status_color = (0, 255, 255)
        elif female_count > 0:
            status = "Female detected - Monitoring"
            status_color = (0, 255, 0)
        elif male_count > 0:
            status = "Male detected - Caution"
            status_color = (255, 255, 0)
        else:
            status = "No people detected"
            status_color = (255, 255, 255)
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Location: {current_location[:40]}...", (10, h-80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Press 'M' for map | 'Q' to quit", (10, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        cv2.imshow('Women Safety Monitor', frame)
        
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

if __name__ == "__main__":
    # Get initial location
    update_location()
    
    # Start background thread for periodic location updates
    location_thread = threading.Thread(
        target=lambda: [update_location() or time.sleep(location_update_interval) for _ in iter(int, 1)],
        daemon=True
    )
    location_thread.start()
    
    main()