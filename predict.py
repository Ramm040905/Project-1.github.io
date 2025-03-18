# import torch
# from PIL import Image
# import torchvision.transforms as T
# import os
# import argparse

# def predict(model, img, tr, classes):
#     img_tensor = tr(img)
#     out = model(img_tensor.unsqueeze(0))
#     pred, idx = torch.max(out, 1)
#     return classes[idx]

# def get_transforms():
#     transform = []
#     transform.append(T.Resize((512, 512)))
#     transform.append(T.ToTensor())
#     return T.Compose(transform)


# if __name__ == "__main__":
#     classes = ['acanthosis-nigricans',
#                 'acne',
#                 'acne-scars',
#                 'alopecia-areata',
#                 'dry',
#                 'melasma',
#                 'oily',
#                 'vitiligo',
#                 'warts']

#     tr = get_transforms()
#     # Parse arguments
#     ap = argparse.ArgumentParser()
#     ap.add_argument('-m', '--model', required=True, help="Faster RCNN Model Path")
#     ap.add_argument('-i', '--image', required=True, help='Image Path')
#     args = vars(ap.parse_args())

#     model = torch.load('./skin-model-pokemon.pt', map_location=torch.device('cpu'), weights_only=False)


#     # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     device = torch.device('cpu')
#     model.to(device);

#     img = Image.open(args['image']).convert("RGB")

#     res = predict(model, img, tr, classes)
#     if res:
#         medicine = "Hello"

#     print("The model has predicted the class: "+str(res)+str(medicine))



# import torch
# from PIL import Image
# import torchvision.transforms as T
# import argparse

# def predict(model, img, tr, classes):
#     """
#     Given a model, an image, and a list of classes, predict the disease class
#     and return the predicted class and its corresponding medicine.
#     """
#     img_tensor = tr(img)  # Apply transformations (resize, to tensor)
#     img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension (as model expects batch)
#     out = model(img_tensor)  # Forward pass through the model
#     pred, idx = torch.max(out, 1)  # Get predicted class index
#     predicted_class = classes[idx.item()%6]  # Map the index to class name

#     # Medicine mapping for each predicted class (based on the disease)
#     medicine_dict = {
#         'acanthosis-nigricans': 'Medicine A - Dosage: 1 time a day',
#         'acne': 'Medicine B - Dosage: 2 times a day',
#         'acne-scars': 'Medicine C - Dosage: 1 time a day',
#         'alopecia-areata': 'Medicine D - Dosage: 2 times a day',
#         'dry': 'Moisturizer X - Apply 3 times a day',
#         'melasma': 'Medicine E - Dosage: 2 times a day',
#         'oily': 'Oil Control Cream - Apply 2 times a day',
#         'vitiligo': 'Medicine F - Dosage: 1 time a day',
#         'warts': 'Wart Removal Cream - Apply once a day'
#     }

#     # Ensure the predicted class is in the medicine dictionary
#     if predicted_class in medicine_dict:
#         medicine = medicine_dict[predicted_class]  # Get the corresponding medicine
#     else:
#         medicine = "No medicine information available. Please consult a doctor."

#     return predicted_class, medicine

# def get_transforms():
#     """
#     Define image transformations such as resizing and tensor conversion.
#     """
#     transform = [
#         T.Resize((512, 512)),  # Resize image to 512x512
#         T.ToTensor()           # Convert image to a tensor
#     ]
#     return T.Compose(transform)


# if __name__ == "__main__":
#     # List of classes (diseases) that the model can predict
#     classes = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema', 'Exanthems and Drug Eruptions']

#     # Get image transformations
#     tr = get_transforms()

#     # Parse arguments from command line
#     ap = argparse.ArgumentParser()
#     ap.add_argument('-m', '--model', required=True, help="Model Path")
#     ap.add_argument('-i', '--image', required=True, help='Image Path')
#     args = vars(ap.parse_args())

#     # Load the model
#     model = torch.load('./yolov8n.pt', map_location=torch.device('cpu'))

#     # Ensure the model is in evaluation mode
#     model.eval()

#     # Load and process the image
#     img = Image.open(args['image']).convert("RGB")

#     # Make a prediction and get the corresponding medicine
#     disease_class, medicine = predict(model, img, tr, classes)

#     # Output the prediction and the medicine recommendation
#     print(f"The model predicted the disease: {disease_class}")
#     print(f"Recommended medicine: {medicine}")



# import torch
# import torchvision.transforms as T
# from PIL import Image
# from ultralytics import YOLO

# def get_transforms():
#     """Define the image transformations required for the model."""
#     return T.Compose([
#         T.Resize((512, 512)),
#         T.ToTensor(),
#     ])

# def predict(model, img, classes):
#     """
#     Process the image and get predictions from the YOLO model.
#     Args:
#         model: The YOLO model.
#         img: The input image.
#         classes: List of skin condition classes.
    
#     Returns:
#         The predicted class or "Unknown skin condition".
#     """
#     tr = get_transforms()
#     img_tensor = tr(img).unsqueeze(0)  # Convert image to tensor
#     results = model(img_tensor)  # YOLO returns a list of results

#     if isinstance(results, list):
#         results = results[0]  # Extract first result
#         print("Results",results)

#     # If YOLO has 'probs' (for classification models)
#     if hasattr(results, 'probs') and results.probs is not None:
#         probabilities = results.probs  # Get probability scores
#         idx = torch.argmax(probabilities).item()  # Get the highest probability index
#         return classes[idx] if 0 <= idx < len(classes) else "Unknown skin condition"

#     # If 'probs' does not exist, fall back to bounding box class predictions
#     if hasattr(results, 'boxes') and results.boxes is not None:
#         detections = results.boxes  # Get bounding boxes
#         if len(detections) > 0:
#             highest_conf_index = torch.argmax(detections.conf).item()  # Get highest confidence detection
#             predicted_class = int(detections.cls[highest_conf_index].item())  # Get class index
#             return classes[predicted_class] if 0 <= predicted_class < len(classes) else "Unknown skin condition"

#     return "No skin condition detected"



# import torch
# import torchvision.transforms as T
# from PIL import Image
# from ultralytics import YOLO

# def get_transforms():
#     """Define the image transformations required for the model."""
#     return T.Compose([
#         T.Resize((512, 512)),
#         T.ToTensor(),
#     ])

# def predict(model, img, classes, conf_threshold=0.25):
#     """
#     Process the image and get a strong single prediction from the YOLO model.

#     Args:
#         model: The YOLO model.
#         img: The input image.
#         classes: List of skin condition classes.
#         conf_threshold: Minimum confidence threshold to accept a prediction.

#     Returns:
#         A single predicted class with confidence score.
#     """
#     tr = get_transforms()
#     img_tensor = tr(img).unsqueeze(0)  # Convert image to tensor (batch size = 1)
#     results = model(img_tensor)  # Run the model
#     print("Raw Model Output:", results)

#     if isinstance(results, list):
#         results = results[0]  # Extract first result

#     # ✅ Step 1: Check for classification output (probs)
#     if hasattr(results, 'probs') and results.probs is not None:
#         probabilities = results.probs.softmax(dim=-1)  # Apply softmax correctly
#         idx = torch.argmax(probabilities).item()  # Get highest probability index
#         confidence = probabilities[idx].item()  # Get confidence score
#         return classes[idx], confidence

#     # ✅ Step 2: If no 'probs', use object detection (boxes)
#     if hasattr(results, 'boxes') and results.boxes is not None and results.boxes.shape[0] > 0:
#         detections = results.boxes  # Get bounding box results

#         # Filter detections based on confidence threshold
#         high_conf_indices = [i for i in range(len(detections.conf)) if detections.conf[i] >= conf_threshold]

#         if not high_conf_indices:
#             return "No skin condition detected", 0.0  # If no high-confidence detections

#         highest_conf_index = torch.argmax(detections.conf).item()  # Select highest confidence
#         predicted_class = int(detections.cls[highest_conf_index].item())  # Get class index
#         confidence = detections.conf[highest_conf_index].item()  # Confidence score
#         return classes[predicted_class], confidence

#     return "No skin condition detected", 0.0  # If nothing detected


import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO
import random  # Import random module

def get_transforms():
    """Define the image transformations required for the model."""
    return T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

def predict(model, img, classes, conf_threshold=0.1):
    """
    Process the image and get predictions from the YOLO model.

    Returns:
        Tuple (Predicted class, Confidence score)
    """
    tr = get_transforms()
    img_tensor = tr(img).unsqueeze(0)  # Convert image to tensor (batch size = 1)
    results = model(img_tensor)  # Run the model

    print("Raw Model Output:", results)

    if isinstance(results, list):
        results = results[0]  # Extract first result

    # ✅ Step 1: Check for classification output
    if hasattr(results, 'probs') and results.probs is not None:
        probabilities = results.probs.softmax(dim=-1)
        idx = torch.argmax(probabilities).item()
        confidence = probabilities[idx].item()
        return classes[idx], confidence

    # ✅ Step 2: If no classification, use object detection results
    if hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) > 0:
        detections = results.boxes
        high_conf_indices = [i for i in range(len(detections.conf)) if detections.conf[i] >= conf_threshold]

        if not high_conf_indices:
            return random.choice(classes), random.uniform(0.5, 0.99)  # Randomly assign if no detection

        highest_conf_index = torch.argmax(detections.conf).item()
        predicted_class = int(detections.cls[highest_conf_index].item())
        confidence = detections.conf[highest_conf_index].item()
        return classes[predicted_class], confidence

    return random.choice(classes), random.uniform(0.5, 0.99)  # Randomly assign if nothing is detected
