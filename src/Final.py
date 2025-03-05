import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import pickle
import requests
import json
from pydub import AudioSegment
from io import BytesIO
import os
import pygame
import locale
locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox, simpledialog, Listbox
import sklearn


# Load the fine-tuned Turkish BERT model, tokenizer, and label encoder
model_path = r"C:\Users\Deniz\Documents\GitHub\Sentiment-Analysis-with-TTS\fine_tuned_results_turkish"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

with open(r"C:\Users\Deniz\Documents\GitHub\Sentiment-Analysis-with-TTS\src\label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Move model to device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

# ElevenLabs API Key (Replace with your own API key)
ELEVENLABS_API_KEY = "sk_6d69377342f29ec53a60f93368ecfb811e8e2edf668589ef"
ELEVENLABS_VOICE_ID = "nPczCjzI2devNBz1zQrb"

def predict_emotion(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1).squeeze().tolist()
    
    predicted_idx = torch.argmax(logits, dim=-1).item()
    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = probabilities[predicted_idx] * 100
    
    return predicted_label, confidence

def speak_with_emotion(sentence):
    url = "https://api.elevenlabs.io/v1/text-to-speech/" + ELEVENLABS_VOICE_ID
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    payload = {
        "text": sentence,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        raise Exception(f"Error in ElevenLabs API: {response.status_code}, {response.text}")

def analyze_text_and_generate_audio(text, filename):
    sentences = text.split(".")
    combined_audio = AudioSegment.empty()
    emotion_results = []
    
    for sentence in sentences:
        if sentence.strip():
            predicted_emotion, confidence = predict_emotion(sentence.strip())
            audio_stream = speak_with_emotion(sentence.strip())
            audio = AudioSegment.from_file(audio_stream, format="mp3")
            combined_audio += audio
            emotion_results.append((sentence.strip(), predicted_emotion, confidence))
    
    combined_audio.export(filename, format="wav")
    return emotion_results

class EmotionDubbingApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="solar")
        self.title("Emotion Dubbing App")
        self.geometry("600x500")
        
        self.text_label = ttk.Label(self, text="Enter Text:", font=("Arial", 12))
        self.text_label.pack(pady=5)
        
        self.text_entry = ttk.Text(self, height=5, width=60)
        self.text_entry.pack(pady=5)
        
        self.convert_button = ttk.Button(self, text="Convert", command=self.convert_text, bootstyle=PRIMARY)
        self.convert_button.pack(pady=5)
        
        self.status_label = ttk.Label(self, text="Status: Ready", font=("Arial", 10), bootstyle=SUCCESS)
        self.status_label.pack(pady=5)
        
        self.audio_file_label = ttk.Label(self, text="WAV File: Not created yet", font=("Arial", 10), bootstyle=INFO)
        self.audio_file_label.pack(pady=5)
        
        self.play_button = ttk.Button(self, text="Play", command=self.play_audio, bootstyle=SUCCESS, state=DISABLED)
        self.play_button.pack(pady=5)
        
        self.stop_button = ttk.Button(self, text="Stop", command=self.stop_audio, bootstyle=DANGER, state=DISABLED)
        self.stop_button.pack(pady=5)
        
        self.emotion_listbox = Listbox(self, height=10, width=75)
        self.emotion_listbox.pack(pady=5)
        
        self.audio_file = None
        self.emotion_results = []
    
    def convert_text(self):
        text = self.text_entry.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showerror("Error", "Please enter text!")
            return
        
        file_name = simpledialog.askstring("File Name", "Enter the name of the audio file:", parent=self)
        if not file_name:
            messagebox.showerror("Error", "Please enter a valid file name!")
            return
        
        save_path = os.path.join(os.getcwd(), f"{file_name}.wav")
        if os.path.exists(save_path):
            messagebox.showerror("Error", f"{file_name}.wav already exists. Choose another name.")
            return
        
        self.status_label.config(text="Status: Processing...", bootstyle=WARNING)
        try:
            self.emotion_results = analyze_text_and_generate_audio(text, save_path)
            self.audio_file = save_path
            self.audio_file_label.config(text=f"WAV File: {self.audio_file}")
            self.status_label.config(text="Status: Conversion Complete", bootstyle=SUCCESS)
            self.play_button.config(state=NORMAL)
            
            self.emotion_listbox.delete(0, "end")
            for sentence, emotion, confidence in self.emotion_results:
                self.emotion_listbox.insert("end", f"{sentence} | {emotion} | {confidence:.2f}%")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.status_label.config(text="Status: Error", bootstyle=DANGER)

    def play_audio(self):
        if self.audio_file:
            pygame.mixer.init()
            pygame.mixer.music.load(self.audio_file)
            pygame.mixer.music.play()
            self.stop_button.config(state=NORMAL)
            self.play_button.config(state=DISABLED)
    
    def stop_audio(self):
        pygame.mixer.music.stop()
        self.stop_button.config(state=DISABLED)
        self.play_button.config(state=NORMAL)
        

if __name__ == "__main__":
    app = EmotionDubbingApp()
    app.mainloop()
