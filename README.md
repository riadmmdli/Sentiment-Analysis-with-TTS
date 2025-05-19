# Emotion Dubbing App 🎙️🧠

Bu proje, kullanıcı tarafından girilen Türkçe metni analiz ederek cümlelerdeki duyguları tahmin eden ve her cümleyi tahmin edilen duyguya göre seslendiren bir **Duygusal Konuşma Üretimi (Emotional Text-to-Speech)** uygulamasıdır.

## 🔍 Özellikler

- 🤖 Türkçe için ince ayarlanmış BERT modeliyle **duygu analizi**
- 🔊 ElevenLabs API kullanılarak **duyguya uygun sesli çıktı**
- 🧾 Cümle bazlı duygu tahmini ve güven skorları
- 💾 WAV formatında çıktı alma ve çalma
- 🖥️ Tkinter + ttkbootstrap ile kullanıcı dostu grafik arayüz

## 🛠️ Kullanılan Teknolojiler

- Python 🐍
- HuggingFace Transformers 🤗
- PyTorch 🔥
- Pydub 🎧
- ElevenLabs API 🎙️
- Tkinter + ttkbootstrap 🎨
- Pygame (ses çalma) 🎼

## 🚀 Başlarken
### Gereksinimler

- Python 3.8+
- Aşağıdaki kütüphaneler:
  ```bash
  pip install torch transformers pydub requests pygame ttkbootstrap
  
## API Anahtarı

ElevenLabs API'yi kullanabilmek için kendi API anahtarınızı almanız gerekmektedir. API anahtarını almak için ElevenLabs web sitesine gidin ve bir hesap oluşturun. Hesabınızı oluşturduktan sonra, API anahtarınızı oluşturabilirsiniz.

Python dosyasına API Anahtarını Ekleyin
Oluşturduğunuz text dosyasının içine API Anahtarınızı ekledikten sonra dosya yolunu aşağıdaki dosya yoluyla değiştirin.
  ```bash
  # Load API key from external file
with open("C:/Users/riadm/Desktop/elevenlabs_api_key.txt", "r") as key_file:
    ELEVENLABS_API_KEY = key_file.read().strip()
  ELEVENLABS_VOICE_ID = "nPczCjzI2devNBz1zQrb"  # Kullanmak istediğiniz sesin ID'si

```
## Model Dosyaları
Bu proje için model dosyalarına ihtiyacınız olacak. Modeli HuggingFace'ten indirip projenize dahil edebilirsiniz. Model dosyalarını aşağıdaki dizine yerleştirmeniz gerekiyor:

- fine_tuned_results_turkish klasörü içerisinde:

- pytorch_model.bin (Model dosyası)

- config.json (Model yapılandırma dosyası)

- vocab.txt (Modelin kelime dağarcığı)

- label_encoder.pkl (Etiketleri çözümlemek için encoder dosyası)

## 📸 Arayüz Görünümü
- Metin giriş alanı

- Duygu analiz ve ses oluşturma butonu

- Oluşturulan WAV dosyasını oynatma ve durdurma butonları

- Duygu, cümle ve güven skoru listesi

## 📂 Çıktı Örneği
Örneğin, aşağıdaki gibi bir çıktı alabilirsiniz:

  ```text
<mutlu> Bugün hava çok güzel. </mutlu> | mutlu | 94.23%
<üzgün> Ama içim biraz buruk. </üzgün> | üzgün | 88.75%
```

## 📌 Notlar
- Her cümle ayrı ayrı analiz edilip seslendirilir.

- Tahmin edilen duygu cümle başı ve sonuna <duygu> etiketleri ile eklenir.

- WAV dosyası olarak dışa aktarılır.

## 🖼️ Ekran Görüntüsü

![image](https://github.com/user-attachments/assets/01f9314b-ed1f-4067-bfe6-0d7446f68864)
