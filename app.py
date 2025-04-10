from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pymongo
import face_recognition
import pygame
import speech_recognition as sr
import gtts
import os
import sounddevice as sd
import numpy as np
import tempfile
from transformers import pipeline
from difflib import SequenceMatcher
import time
import threading
from datetime import datetime , timedelta
from gtts import gTTS
import secrets
import cv2
import uuid
import webbrowser
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import math
from collections import Counter
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['UPLOAD_FOLDER_IMAGES'] = 'static/images'

# MongoDB Connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["Neuroassist-Companion"]
collection_patients = db["patients"]
collection_prompts = db["prompts"]
collection_family = db["family_members"]
collection_reminders = db["reminders"]
collection_interactions = db["patient_interact"]
collection_prompt_history = db["prompt_history"]
collection_memories = db["memories"]


hugging_face_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/patient')
def patient():
    return render_template('patient.html')

@app.route("/caregiver")
def caregiver_home():
    return render_template("caregiver.html")

@app.route("/add_prompt", methods=["GET", "POST"])
def add_prompt():
    if request.method == "POST":
        patient_name = request.form["patient_name"]
        caregiver_name = request.form["caregiver_name"]
        age = int(request.form["age"])
        new_patient = {"patient_name": patient_name, "caregiver_name": caregiver_name, "age": age}
        collection_patients.insert_one(new_patient)
        return render_template("caregiver_prompt_response.html", caregiver_name=caregiver_name, patient_name=patient_name)
    return render_template("caregiver_add_prompt.html")

@app.route("/caregiver/prompt_response/<caregiver_name>/<patient_name>", methods=["GET", "POST"])
def caregiver_prompt_response(caregiver_name, patient_name):
    if request.method == "POST":
        prompt = request.form["prompt"]
        response = request.form["response"]
        new_prompt = {"patient_name": patient_name, "prompt": prompt, "response": response}
        collection_prompts.insert_one(new_prompt)
        return render_template("caregiver_prompt_response.html", caregiver_name=caregiver_name, patient_name=patient_name, success_message="Prompt and response added successfully!")
    return render_template("caregiver_prompt_response.html", caregiver_name=caregiver_name, patient_name=patient_name)

   


def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=True) as temp_audio:
        temp_audio_path = temp_audio.name + ".mp3"
        tts.save(temp_audio_path)
        os.system(f"start {temp_audio_path}")

def fetch_response_from_db(prompt, patient_name):
    all_prompts = list(collection_prompts.find({"patient_name": patient_name}, {"prompt": 1, "response": 1}))
    best_match = None
    highest_similarity = 0.6
    for doc in all_prompts:
        similarity = SequenceMatcher(None, doc["prompt"].lower(), prompt.lower()).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = doc["response"]
    return best_match

# Check if prompt is music-related
def is_music_command(prompt):
    keywords = ["play music", "play some music", "play songs", "music please", "i want music"]
    for keyword in keywords:
        if keyword in prompt.lower():
            return True
    return False

# Open Spotify search in browser (free account support)
def open_spotify_search(query):
    search_url = f"https://open.spotify.com/search/{query.replace(' ', '%20')}"
    webbrowser.open(search_url)

@app.route("/start_session", methods=["POST"])
def start_session():
    patient_name = request.form.get("patient_name")
    if patient_name:
        session["patient_name"] = patient_name
        return jsonify({"message": f"Session started for {patient_name}"})
    return jsonify({"error": "No patient name provided!"})

@app.route("/voice_assistant")
def voice_assistant():
    return render_template("voice_assistant.html")

@app.route("/process_voice", methods=["POST"])
def process_voice():
    if "patient_name" not in session:
        return jsonify({"error": "No patient session found!"})
    patient_name = session["patient_name"]
    data = request.json
    prompt = data.get("prompt")
    
    # Handle music command without premium
    if is_music_command(prompt):
        speak("Opening Spotify for you")
        open_spotify_search("relaxing music")  # You can customize this
        return jsonify({"response": "Opened Spotify for relaxing music"})# Handle music command without premium
    if is_music_command(prompt):
        speak("Opening Spotify for you")
        open_spotify_search("relaxing music")  # You can customize this
        return jsonify({"response": "Opened Spotify for relaxing music"})
    
    response = fetch_response_from_db(prompt, patient_name)
    if not response:
        response = hugging_face_pipeline(prompt, max_length=50)[0]['generated_text']
    speak(response)
    
    # 💾 Save prompt & response to history (NO timestamp)
    collection_interactions.insert_one({
        "patient_name": patient_name,
        "prompt": prompt,
        "response": response
    })
    
    return jsonify({"response": response})
 
@app.route("/chat_assistant_page")
def chat_assistant_page():
    return render_template("chat_assistant.html")


@app.route("/chat_assistant", methods=["POST"])
def chat_assistant():
    data = request.get_json()
    prompt = data.get("prompt")
    patient_name = data.get("patient_name")

    if not prompt or not patient_name:
        return jsonify({"error": "Invalid input!"}), 400

    # 1. Check DB first
    response = fetch_response_from_db(prompt, patient_name)

    # 2. Check if it's a music command
    if is_music_command(prompt):
        return jsonify({"music_redirect": True})

    # 3. Use Hugging Face if no match found
    if not response:
        response = hugging_face_pipeline(prompt)[0]["generated_text"]

    # 4. Store prompt and response to history
    collection_prompt_history.insert_one({
        "patient_name": patient_name,
        "prompt": prompt,
        "response": response,
        "chat_id": str(uuid.uuid4()),
        "timestamp": datetime.now()
    })

    return jsonify({"response": response})

@app.route("/chat_history", methods=["GET"])
def chat_history():
    patient_name = request.args.get("patient_name")
    today_chats = []
    yesterday_chats = []

    if patient_name:
        chats_cursor = collection_prompt_history.find(
            {"patient_name": patient_name}
        ).sort("timestamp", 1)

        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        for chat in chats_cursor:
            prompt = chat.get("prompt")
            response = chat.get("response")
            timestamp = chat.get("timestamp")
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "N/A"

            if timestamp:
                chat_date = timestamp.date()
                entry = {
                    "prompt": prompt,
                    "response": response,
                    "timestamp": formatted_time
                }

                if chat_date == today:
                    today_chats.append(entry)
                elif chat_date == yesterday:
                    yesterday_chats.append(entry)

    return render_template("chat_history.html", 
                           today_chats=today_chats, 
                           yesterday_chats=yesterday_chats, 
                           patient_name=patient_name)



@app.route('/caregiver_upload', methods=['GET', 'POST'])
def caregiver_upload():
    if request.method == 'POST':
        patient_name = request.form['patient_name']
        member_name = request.form['member_name']
        relationship = request.form['relationship']
        image = request.files['image']
        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            new_family_member = {
                "patient_name": patient_name,
                "member_name": member_name,
                "relationship": relationship,
                "image_path": image_path
            }
            collection_family.insert_one(new_family_member)
            return render_template("caregiver_upload.html", success_message="Image uploaded successfully!")
    return render_template("caregiver_upload.html")

@app.route('/caregiver_reminders', methods=['GET', 'POST'])
def caregiver_reminders():
    if request.method == 'POST':
        patient_name = request.form['patient_name']
        reminder_text = request.form['reminder_text']
        reminder_time = request.form['reminder_time']
        repeat = request.form['repeat']  # "none", "daily", "weekly"
        notify_type = request.form['notify_type']  # "voice", "popup", "both"
        try:
            full_reminder_time = datetime.strptime(reminder_time, "%Y-%m-%dT%H:%M")
        except ValueError:
            return render_template("caregiver_reminders.html", error_message="Invalid time format!")

        
        new_reminder = {
            "patient_name": patient_name,
            "reminder_text": reminder_text,
            "reminder_time": full_reminder_time.strftime('%H:%M'),
            "repeat": repeat,
            "notify_type": notify_type
        }
        collection_reminders.insert_one(new_reminder)
        return render_template("caregiver_reminders.html", success_message="Reminder Set Successfully!")
    return render_template('caregiver_reminders.html')

def check_reminders():
    while True:
        # Get current time up to minute precision
        now = datetime.now().strftime('%H:%M')

        # Find reminders scheduled for current time
        reminders = list(collection_reminders.find({
            "reminder_time": now
        }))

        for reminder in reminders:
            text = reminder["reminder_text"]
            notify_type = reminder["notify_type"]
            repeat = reminder["repeat"]

            # Voice Notification
            if notify_type in ["voice", "both"]:
                speak(text)
                
            # Pop-up Notification (console for now)
            if notify_type in ["popup", "both"]:
                print(f"🔔 POP-UP: {text}")

            # Repeat Handling
            if repeat in ["daily", "weekly"]:
                next_time = datetime.now()
                if repeat == "daily":
                    next_time += timedelta(days=1)
                elif repeat == "weekly":
                    next_time += timedelta(weeks=1)

                next_reminder_time = next_time.strftime('%H:%M')
                collection_reminders.update_one(
                    {"_id": reminder["_id"]},
                    {"$set": {"reminder_time": next_reminder_time}}
                )
            else:
                collection_reminders.delete_one({"_id": reminder["_id"]})

        time.sleep(60)  # Wait 1 minute before next check


reminder_thread = threading.Thread(target=check_reminders, daemon=True)
reminder_thread.start()

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/history", methods=["GET", "POST"])
def history():
    patient = None
    history = []

    if request.method == "POST":
        patient_name = request.form.get("patient_name")

        # Get patient details
        patient = collection_patients.find_one({"patient_name": patient_name})

        # Get prompt/response history
        if patient:
            history = list(collection_interactions.find(
                {"patient_name": patient_name}
            ))

    return render_template("history.html", patient=patient, history=history)

@app.route("/charts-insights", methods=["GET","POST"])
def charts_insights():
    patient_name = None
    stats = {}
    dates = []
    counts = []
    words = []

    if request.method == "POST":
        patient_name = request.form.get("patient_name")

        interactions = list(collection_interactions.find({"patient_name": patient_name}))

        if interactions:
            date_counter = Counter()
            all_words = []

            for item in interactions:
                ts = item.get("timestamp")
                try:
                    date = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").date()
                    date_counter[date] += 1
                except:
                    continue

                prompt_text = item.get("prompt", "").lower()
                for word in prompt_text.split():
                    if word.isalpha() and word not in ["what", "is", "the", "a", "an", "to", "and", "of", "in"]:
                        all_words.append(word)

            sorted_dates = sorted(date_counter.items())
            dates = [str(d[0]) for d in sorted_dates]
            counts = [d[1] for d in sorted_dates]

            word_freq = Counter(all_words).most_common(10)
            words = [{"text": w, "value": c} for w, c in word_freq]

            total = len(interactions)
            unique_days = len(set(dates))
            avg_per_day = round(total / unique_days, 2) if unique_days else total
            
            # ✅ FIX: check if date_counter is not empty
            most_active_day = str(max(date_counter, key=date_counter.get)) if date_counter else "N/A"
            

            stats = {
                "total": total,
                "avg_per_day": avg_per_day,
                "most_active_day": str(most_active_day)
            }

    return render_template("charts_insights.html", dates=dates, counts=counts, words=words, stats=stats, patient_name=patient_name)


@app.route("/memory_album")
def memory_album():
    images = list(collection_memories.find().sort("timestamp", -1))
    return render_template("memory_album.html", images=images)

@app.route("/upload-memory", methods=["POST"])
def upload_memory():
    patient_name = request.form['patient_name']
    relationship = request.form['relationship']
    description = request.form['description']
    image = request.files['image']

    if image:
        filename = secure_filename(image.filename)
        upload_folder = app.config['UPLOAD_FOLDER_IMAGES']
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, filename)
        image.save(filepath)

        memory_data = {
            "patient_name": patient_name,
            "relationship": relationship,
            "description": description,
            "image_path": filepath,
            "timestamp": datetime.now()
        }

        collection_memories.insert_one(memory_data)
        

    return redirect(url_for('memory_album', success_message='Memory uploaded successfully!'))





    


    
# Load known faces at startup
known_face_encodings = []
known_face_names = []
relationship_dict = {}

family_members = list(collection_family.find({}))
for member in family_members:
    image_path = member['image_path']
    if os.path.exists(image_path):
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(member["member_name"])
            relationship_dict[member["member_name"]] = member["relationship"]

@app.route("/face_recognition")
def face():
    return render_template("face_recognition.html")

def speak(text):
    tts = gTTS(text)
    filename = f"voice_{uuid.uuid4().hex}.mp3"
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, filename)
    tts.save(temp_audio_path)
    
    # Play the audio (Windows)
    os.system(f'start "" "{temp_audio_path}"')

@app.route("/recognize_face")
def recognize_face():
    # Reload encodings fresh every time
    known_face_encodings = []
    known_face_names = []
    relationship_dict = {}

    family_members = list(collection_family.find({}))
    for member in family_members:
        image_path = member['image_path']
        if image_path and os.path.exists(image_path):
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(member["member_name"])
                relationship_dict[member["member_name"]] = member["relationship"]
    if not known_face_encodings:
        return jsonify({"message": "No known faces loaded!"})


    # Access camera and recognize
    video_capture = cv2.VideoCapture(0)
    
    ret, frame = video_capture.read()
    video_capture.release()
    if not ret:
        return jsonify({"message": "Failed to access camera!"})

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches and matches[best_match_index]:
            name = known_face_names[best_match_index]
            relationship = relationship_dict.get(name, "unknown relation")
            response_text = f"This is {name}, your {relationship}."

            speak(response_text)
            video_capture.release()

            return jsonify({"message": response_text})
                
    video_capture.release()
    return jsonify({"message": "Face not recognized!"})

if __name__ == '__main__':
    app.run(debug=True)