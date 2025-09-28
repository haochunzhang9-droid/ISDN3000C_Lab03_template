# classify_batch.py
#
# FOR THE ADVANCED TASK
# This script processes all images in a specified folder, runs inference on each,
# and saves the results to a CSV file.
#
# Usage:
# python classify_folder.py <path_to_image_folder>
# classify_batch.py
#
# Advanced Task: Batch image classification
# Usage:
#   python classify_batch.py <path_to_image_fold# classify_batch.py
#
# Advanced Task: Batch image classification
# Usage:
#   python classify_batch.py <path_to_image_folder>

import os
import sys
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import csv
import urllib.request

def load_model():
    print("Loading pre-trained model: ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    print("Model loaded successfully.")
    return model

def load_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    print("Downloading class labels...")
    labels = json.load(urllib.request.urlopen(url))
    print("Labels loaded.")
    return labels

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def classify_images(folder_path, model, labels):
    results = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            input_tensor = preprocess_image(fpath)
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = outputs.max(1)
                label = labels[predicted.item()]
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item() * 100
                results.append([fname, label, f"{confidence:.2f}%"])
                print(f"{fname}: {label} ({confidence:.2f}%)")
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    return results

def save_results(results):
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "detected_class", "confidence_level"])
        writer.writerows(results)
    print("Results saved to results.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_batch.py <path_to_image_folder>")
        sys.exit(1)

    folder = sys.argv[1]
    model = load_model()
    labels = load_labels()
    results = classify_images(folder, model, labels)
    save_results(results)

