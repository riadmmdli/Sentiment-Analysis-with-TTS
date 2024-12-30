import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import Trainer, TrainingArguments

# Emotion labels
EMOTION_LABELS = ["Happy", "Fear", "Anger", "Sadness", "Disgust", "Surprise"]

# Step 1: Load and preprocess the dataset
file_path = "C:/Users/riadm/Desktop/BertTurkModelFinal/data/set_cleaned_data.csv"  # Replace with your dataset path
df = pd.read_csv(file_path)
df_clean = df.dropna(subset=['Entry', 'ValidatedEmotion'])

# Ensure only the specified emotion labels are used
df_clean = df_clean[df_clean['ValidatedEmotion'].isin(EMOTION_LABELS)]

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(EMOTION_LABELS)  # Use the specified emotion labels
df_clean['EncodedEmotion'] = label_encoder.transform(df_clean['ValidatedEmotion'])

# Split dataset into train and test sets
X = df_clean['Entry'].tolist()
y = df_clean['EncodedEmotion'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Tokenize the text data
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# Create Dataset class
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, y_train)
test_dataset = EmotionDataset(test_encodings, y_test)

# Step 3: Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('C:/Users/riadm/Desktop/BertTurkModelFinal/fine_tuned_results_turkish')
tokenizer = BertTokenizer.from_pretrained('C:/Users/riadm/Desktop/BertTurkModelFinal/fine_tuned_results_turkish')

# Load the label encoder
with open('C:/Users/riadm/Desktop/BertTurkModelFinal/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Move model to device (CPU or GPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

# Step 4: Define the Trainer and evaluation arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=64,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset
)

# Step 5: Get predictions for the entire test set
predictions, labels, _ = trainer.predict(test_dataset)
predicted_labels = torch.argmax(torch.tensor(predictions), dim=-1)

# Ensure that both labels and predictions are on the same device (CPU)
labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
predicted_labels = predicted_labels.cpu().numpy() if isinstance(predicted_labels, torch.Tensor) else predicted_labels

# Step 6: Calculate accuracy
accuracy = accuracy_score(labels, predicted_labels) * 100  # Convert to percentage
print(f"Model Accuracy: {accuracy:.2f}%")

# Step 7: Classification report
report = classification_report(labels, predicted_labels, target_names=EMOTION_LABELS)
print("Classification Report:")
print(report)

# Step 8: Confusion Matrix
cm = confusion_matrix(labels, predicted_labels)

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Step 9: Accuracy per Emotion (Bar Chart)
emotion_accuracy = {}
for i, label in enumerate(EMOTION_LABELS):
    correct = np.sum((labels == i) & (predicted_labels == i))
    total = np.sum(labels == i)
    emotion_accuracy[label] = correct / total if total != 0 else 0

# Plot accuracy for each emotion
plt.figure(figsize=(10, 6))
sns.barplot(x=list(emotion_accuracy.keys()), y=list(emotion_accuracy.values()))
plt.title('Accuracy per Emotion')
plt.ylabel('Accuracy')
plt.xlabel('Emotion')
plt.ylim(0, 1)
plt.show()
