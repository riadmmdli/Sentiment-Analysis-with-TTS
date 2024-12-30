import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd
import pickle
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

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

# Save the label encoder for future use
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Split dataset into train and test sets
X = df_clean['Entry'].tolist()
y = df_clean['EncodedEmotion'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# Create Dataset class
class EmotionDataset(Dataset):
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

# Step 2: Load the pre-trained model for fine-tuning
model = BertForSequenceClassification.from_pretrained(
    'dbmdz/bert-base-turkish-cased',
    num_labels=len(EMOTION_LABELS)
)

# Step 3: Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    save_total_limit=2
)

# Step 4: Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()

# Step 5: Evaluate the fine-tuned model manually for accuracy
predictions, labels, _ = trainer.predict(test_dataset)
predicted_labels = torch.argmax(torch.tensor(predictions), dim=-1)

# Move labels and predictions to CPU if they are on GPU
if isinstance(predicted_labels, torch.Tensor):
    predicted_labels = predicted_labels.cpu().numpy()
if isinstance(labels, torch.Tensor):
    labels = labels.cpu().numpy()

# Calculate accuracy using sklearn's accuracy_score
accuracy = accuracy_score(labels, predicted_labels) * 100  # Convert to percentage
print(f"Model Accuracy: {accuracy:.2f}%")

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine_tuned_results_turkish')
tokenizer.save_pretrained('./fine_tuned_results_turkish')
print("Fine-tuning complete. Model saved to './fine_tuned_results_turkish'")

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('./fine_tuned_results_turkish')
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_results_turkish')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Move model to device (CPU or GPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

def predict_emotion(sentence):
    """
    Predicts the emotion of a given sentence using the trained model.
    """
    # Tokenize and encode the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to the same device as the model
 
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculate probabilities
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1).squeeze().tolist()

    # Get the predicted label and confidence
    predicted_idx = torch.argmax(logits, dim=-1).item()
    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = probabilities[predicted_idx] * 100

    # Return the prediction
    return predicted_label, confidence

# Allow the user to input sentences and get predictions
while True:
    user_input = input("\nEnter sentences in Turkish separated by a period (or type 'exit' to quit): ")
    sentences = user_input.split(".")  # Split input into multiple sentences

    for sentence in sentences:
        sentence = sentence.strip()  # Remove any extra spaces
        if sentence:
            predicted_emotion, confidence = predict_emotion(sentence)
            print(f"\nSentence: {sentence}\nPredicted Emotion: {predicted_emotion} (Confidence: {confidence:.2f}%)")


   