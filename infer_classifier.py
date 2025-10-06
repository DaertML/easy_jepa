"""
Script for image classification inference using trained model.

USAGE:
python infer_classifier.py --weights <path to the weights.pth file> \
--input <directory containing inference images>

Update the `CLASS_NAMES` list to contain the trained class names. 
"""

import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import glob
import argparse
import pathlib

from src.img_cls.model import LinearClassifier

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-w', '--weights', 
    default='../outputs/best_model.pth',
    help='path to the model weights',
)
parser.add_argument(
    '--input',
    required=True,
    help='directory containing images for inference'
)
args = parser.parse_args()

# Constants and other configurations.
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 256
CLASS_NAMES = [
    'Healthy', 'Powdery', 'Rust'
]

CLASS_NAMES = ['bag', 'dress', 'hoodie', 'shoe', 'sneaker', 'trousers', 'tshirt']

CLASS_NAMES = ['FIRE', 'LEFT', 'NOOP', 'RIGHT']

# Validation transforms
def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    return test_transform

def annotate_image(output_class, orig_image):
    class_name = CLASS_NAMES[int(output_class)]
    cv2.putText(
        orig_image, 
        f"{class_name}", 
        (5, 35), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (0, 0, 255), 
        2, 
        lineType=cv2.LINE_AA
    )
    return orig_image

def inference(model, testloader, device, orig_image):
    """
    Function to run inference.

    :param model: The trained model.
    :param testloader: The test data loader.
    :param DEVICE: The computation device.
    """
    model.eval()
    counter = 0

    with torch.no_grad():
        counter += 1
        image = testloader
        image = image.to(device)

        # Forward pass.
        outputs = model(image)

    # Softmax probabilities.
    predictions = F.softmax(outputs, dim=1).cpu().numpy()
    # Predicted class number.
    output_class = np.argmax(predictions)
    # Show and save the results.
    result = annotate_image(output_class, orig_image)
    return result

if __name__ == '__main__':
    weights_path = pathlib.Path(args.weights)
    infer_result_path = os.path.join(
        'outputs', 'inference_results'
    )
    os.makedirs(infer_result_path, exist_ok=True)

    checkpoint = torch.load(weights_path)
    # Load the model.
    model = LinearClassifier(
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    all_image_paths = glob.glob(os.path.join(args.input, '*'))

    transform = get_test_transform(IMAGE_RESIZE)

    for i, image_path in enumerate(all_image_paths):
        print(f"Inference on image: {i+1}")
        image = cv2.imread(image_path)
        orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        result = inference(
            model, 
            image,
            DEVICE,
            orig_image
        )
        # Save the image to disk.
        image_name = image_path.split(os.path.sep)[-1]
        # cv2.imshow('Image', result)
        # cv2.waitKey(1)
        cv2.imwrite(
            os.path.join(infer_result_path, image_name), result
        )
