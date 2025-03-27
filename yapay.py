import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Veri yolları
json_path = r"output_data.json"  
image_folder = r"OutputImages"

# Veri setini yükleme
class SignLanguageDataset(Dataset):
    def __init__(self, json_path, image_size=(64, 64)):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_paths = []
        self.labels = []
        self.label_map = {word: idx for idx, word in enumerate(self.data.keys())}  

        for word, frames in self.data.items():
            for frame in frames:
                img_path = os.path.normpath(os.path.join(image_folder, os.path.basename(frame["image_path"])))

                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, image_size)  
                        img = img / 255.0  
                        self.image_paths.append(img)
                        self.labels.append(self.label_map[word])
                    else:
                        print(f"Resim yüklenemedi: {img_path}")
                else:
                    print(f"Dosya mevcut değil: {img_path}")

        self.image_paths = np.array(self.image_paths, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.image_paths[idx]
        label = self.labels[idx]
        img = np.transpose(img, (2, 0, 1))  
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return img, label

# Veri yükleme
dataset = SignLanguageDataset(json_path)

if len(dataset) == 0:
    raise ValueError("Veri kümesi boş! Lütfen JSON dosyasını ve resim dosyalarını kontrol edin.")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# **Yapay Sinir Ağı (ANN) Modeli**
class ANNModel(nn.Module):
    def __init__(self, num_classes):
        super(ANNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 64 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Model oluşturma
num_classes = len(dataset.label_map)
model = ANNModel(num_classes)

# Eğitim için kriter ve optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Model eğitme
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Modeli kaydetme
torch.save(model.state_dict(), 'ann_sign_language_model.pth')

# **Test Aşaması: Accuracy Hesaplama**
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Modeli yükleme
model = ANNModel(num_classes)
model.load_state_dict(torch.load('ann_sign_language_model.pth'))
model.eval()

# **Kelime Tahmini Fonksiyonu**
def predict_word(image_path, model, label_map, image_size=(64, 64)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted_label = torch.max(outputs, 1)

    for word, idx in label_map.items():
        if idx == predicted_label:
            return word
    return None

# **Video Oluşturma Fonksiyonu (Yapay Sinir Ağı ile)**
def generate_video(sentence, model, label_map, data, output_video="yapay_sinir_agi.mp4", fps=15):
    words = sentence.lower().split()
    frame_list = []
    target_resolution = (256, 256)  

    for word in words:
        if word in data:
            for frame in data[word]:
                img_path = os.path.normpath(os.path.join(image_folder, os.path.basename(frame["image_path"])))

                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_CUBIC)
                        frame_list.append(img)
                    else:
                        print(f"Resim yüklenemedi: {img_path}")
                else:
                    print(f"Dosya mevcut değil: {img_path}")

    if not frame_list:
        print("Hiçbir kare eklenemedi!")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    out = cv2.VideoWriter(output_video, fourcc, fps, target_resolution)

    for frame in frame_list:
        out.write(frame)

    out.release()
    print(f"Video kaydedildi: {output_video}")

# Kullanıcıdan cümle al
sentence = "merhaba ben universite ogrenciyim"
generate_video(sentence, model, dataset.label_map, dataset.data)
