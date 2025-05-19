import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_scheduler
)
from tqdm import tqdm

# 1. Load and prepare dataset
df = pd.read_csv("C:/Users/riadm/Desktop/BertTurkModelFinal/data/set_cleaned_data.csv")
df = df[["Entry", "ValidatedEmotion"]].dropna()

# 2. Encode target labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['ValidatedEmotion'])

# 3. Tokenizer
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

# 4. Custom Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 5. Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    df['Entry'].tolist(), df['label'].tolist(), test_size=0.1, random_state=42)

train_dataset = EmotionDataset(X_train, y_train, tokenizer)
val_dataset = EmotionDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 6. Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased",
    num_labels=len(label_encoder.classes_)
)
model.to(device)

# 7. Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_training_steps = len(train_loader) * 3
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=num_training_steps)

# 8. Training Loop with tqdm
model.train()
for epoch in range(3):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loop.set_postfix(loss=loss.item())

# 9. Evaluation with tqdm
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

# 10. Metrics
target_names = label_encoder.classes_
report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
print("Classification Report:\n", report)

# 11. Confusion Matrix
conf_mat = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 12. Save model and encoder
model.save_pretrained("output_model/")
tokenizer.save_pretrained("output_model/")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
