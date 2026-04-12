import torch
import json
import argparse
from torchvision import models
from torch import nn
from utils import process_image

def load_model(checkpoint_path):
    model = models.vgg16(pretrained=True)

    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    return model

def predict(image_path, model, topk=5):
    image = process_image(image_path).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = torch.exp(output)

    top_probs, top_classes = probs.topk(topk)
    return top_probs[0].tolist(), top_classes[0].tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()

    model = load_model("model/checkpoint.pth")

    probs, classes = predict(args.image_path, model)

    print("Top Predictions:")
    for p, c in zip(probs, classes):
        print(f"Class: {c}, Probability: {p:.4f}")
