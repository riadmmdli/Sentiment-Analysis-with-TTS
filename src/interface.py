import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import pickle
from TTS.api import TTS  # Import Coqui TTS
from pydub import AudioSegment
import tkinter as tk
from tkinter import messagebox, simpledialog
import os
import pygame

# Load the fine-tuned Turkish BERT model, tokenizer, and label encoder
model = BertForSequenceClassification.from_pretrained('C:/Users/riadm/Desktop/BertTurkModelFinal/fine_tuned_results_turkish')
tokenizer = BertTokenizer.from_pretrained('C:/Users/riadm/Desktop/BertTurkModelFinal/fine_tuned_results_turkish')

with open('C:/Users/riadm/Desktop/BertTurkModelFinal/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load a Turkish TTS model from Coqui
tts = TTS(model_name="tts_models/tr/common-voice/glow-tts", progress_bar=False)

# Move BERT model to device (CPU or GPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

# Predict Emotion
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

# Speak with emotion
def speak_with_emotion(sentence, emotion, filename):
    emotion_styles = {
        "mutluluk": {"speed": 1.2, "pitch": 1.2},
        "üzüntü": {"speed": 0.8, "pitch": 0.8},
        "öfke": {"speed": 1.0, "pitch": 1.3},
        "korku": {"speed": 0.9, "pitch": 0.9},
        "şaşkınlık": {"speed": 1.3, "pitch": 1.1},
        "nötr": {"speed": 1.0, "pitch": 1.0},
    }

    style = emotion_styles.get(emotion, {"speed": 1.0, "pitch": 1.0})
    tts.tts_to_file(
        text=sentence,
        file_path=filename,
        speed=style["speed"],
        pitch=style["pitch"]
    )

# Analyze Text and create combined audio
def analyze_text_and_generate_audio(text, filename):
    sentences = text.split(".")
    combined_audio = AudioSegment.empty()

    emotion_results = []  # To store emotion predictions for each sentence

    for sentence in sentences:
        if sentence.strip():
            predicted_emotion, confidence = predict_emotion(sentence.strip())
            audio_file = filename
            speak_with_emotion(sentence.strip(), predicted_emotion, audio_file)
            audio = AudioSegment.from_wav(audio_file)
            combined_audio += audio
            
            # Store emotion prediction and confidence for display
            emotion_results.append((sentence.strip(), predicted_emotion, confidence))

    combined_audio.export(filename, format="wav")

    return emotion_results

# Tkinter Interface
class EmotionDubbingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Duygu Dubbing Uygulaması")
        self.root.geometry("600x500")
        
        # Metin giriş kutusu
        self.text_label = tk.Label(root, text="Metni Girin:")
        self.text_label.pack(pady=10)
        
        self.text_entry = tk.Text(root, height=5, width=50)
        self.text_entry.pack(pady=10)
        
        # Dönüştür butonu
        self.convert_button = tk.Button(root, text="Dönüştür", command=self.convert_text)
        self.convert_button.pack(pady=10)

        # Durum etiketi
        self.status_label = tk.Label(root, text="Durum: Hazır", fg="green")
        self.status_label.pack(pady=10)

        # Ses dosyası gösterimi
        self.audio_file_label = tk.Label(root, text="WAV dosyası: Henüz oluşturulmadı", fg="blue")
        self.audio_file_label.pack(pady=10)

        # Ses oynatma butonları
        self.play_button = tk.Button(root, text="Oynat", command=self.play_audio, state=tk.DISABLED)
        self.play_button.pack(pady=5)
        
        self.stop_button = tk.Button(root, text="Durdur", command=self.stop_audio, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        # Duygu ve doğruluk oranlarını göstermek için etiket
        self.emotion_label = tk.Label(root, text="Tahmin edilen duygular ve doğruluk oranları:", anchor="w")
        self.emotion_label.pack(pady=10)

        self.emotion_listbox = tk.Listbox(root, width=70, height=10)
        self.emotion_listbox.pack(pady=10)

        # Ses dosyası
        self.audio_file = None
        self.emotion_results = []

    def convert_text(self):
        text = self.text_entry.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showerror("Hata", "Lütfen metin giriniz!")
            return
        
        # Kullanıcıdan dosya adı alma
        file_name = simpledialog.askstring("Dosya Adı", "Lütfen ses dosyasının adını girin:", parent=self.root)
        
        if not file_name:
            messagebox.showerror("Hata", "Lütfen geçerli bir dosya adı giriniz!")
            return
        
        # Dosya adı kontrolü (Visual Studio Code çalışma dizini)
        save_path = os.path.join(os.getcwd(), f"{file_name}.wav")  # Kaydetme yolunu belirliyoruz
        if os.path.exists(save_path):
            messagebox.showerror("Hata", f"{file_name}.wav adıyla zaten bir dosya var. Lütfen başka bir isim girin.")
            return
        
        self.status_label.config(text="Durum: İşleniyor...", fg="orange")
        
        # Metni sesli duygu analizi ve dönüşüm işlemi
        try:
            self.emotion_results = analyze_text_and_generate_audio(text, save_path)
            self.audio_file = save_path
            self.audio_file_label.config(text=f"WAV dosyası: {self.audio_file}")
            self.status_label.config(text="Durum: Dönüşüm Tamamlandı", fg="green")
            self.play_button.config(state=tk.NORMAL)
            
            # Duygu sonuçlarını listede göster
            self.emotion_listbox.delete(0, tk.END)  # Clear existing list
            for sentence, emotion, confidence in self.emotion_results:
                self.emotion_listbox.insert(tk.END, f"Cümle: {sentence} | Duygu: {emotion} | Güven: {confidence:.2f}%")
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            self.status_label.config(text="Durum: Hata", fg="red")

    def play_audio(self):
        if self.audio_file:
            pygame.mixer.init()
            pygame.mixer.music.load(self.audio_file)
            pygame.mixer.music.play()
            self.stop_button.config(state=tk.NORMAL)
            self.play_button.config(state=tk.DISABLED)
            pygame.mixer.music.set_endevent(pygame.USEREVENT)
            self.root.after(100, self.check_audio_status)

    def check_audio_status(self):
        for event in pygame.event.get():
            if event.type == pygame.USEREVENT:
                self.stop_button.config(state=tk.DISABLED)
                self.play_button.config(state=tk.NORMAL)

    def stop_audio(self):
        pygame.mixer.music.stop()
        self.stop_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.NORMAL)

# Ana pencereyi oluştur
root = tk.Tk()
app = EmotionDubbingApp(root)

# Uygulamayı başlat
root.mainloop()
