# import streamlit as st
# import torch
# from PIL import Image
# from torchvision import models, transforms
# import torch.nn as nn

# # Load the trained model
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 2)  # For binary classification
# model.load_state_dict(torch.load('madhubani_classifier.pth'))
# model.eval()

# # Define the transformation for the input image
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize all images to 224x224
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
# ])

# # Define the function to make predictions
# def predict_image(image, model, transform):
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     with torch.no_grad():
#         output = model(image)
#         _, predicted = output.max(1)
#     return "Madhubani" if predicted.item() == 0 else "Not Madhubani"

# # Streamlit app layout
# st.title("Madhubani Painting Classifier")
# st.write("Upload an image of a painting to check if it's Madhubani style or not.")

# # File uploader widget
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Open and display the image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image.", use_column_width=True)
    
#     # Make the prediction
#     prediction = predict_image(image, model, transform)
#     st.write(f"Prediction: {prediction}")

# import streamlit as st

# st.title("Madhubani Painting Classifier")
# st.write("Upload an image of a painting to check if it's Madhubani style or not.")

# # File uploader widget
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
#     st.write("File successfully uploaded!")
# else:
#     st.write("Please upload an image to continue.")

# import streamlit as st
# import torch
# from PIL import Image
# from torchvision import transforms

# # Streamlit app layout
# st.title("Madhubani Painting Classifier")
# st.write("Upload an image of a painting to check if it's Madhubani style or not.")

# # File uploader widget
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Open and display the image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image.", use_column_width=True)

#     # Temporary check (without model prediction part)
#     st.write("Image uploaded! Next step is prediction...")

# import streamlit as st
# import torch
# from PIL import Image
# from torchvision import models, transforms
# import torch.nn as nn

# # Load the trained model
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 2)  # For binary classification
# model.load_state_dict(torch.load('madhubani_classifier.pth'))
# model.eval()

# # Define the transformation for the input image
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize all images to 224x224
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
# ])

# # Define the function to make predictions
# def predict_image(image, model, transform):
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     with torch.no_grad():
#         output = model(image)
#         _, predicted = output.max(1)
#     return "Madhubani" if predicted.item() == 0 else "Not Madhubani"

# # Streamlit app layout
# st.title("Madhubani Painting Classifier")
# st.write("Upload an image of a painting to check if it's Madhubani style or not.")

# # File uploader widget
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Open and display the image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image.", use_column_width=True)

#     # Add a button for prediction
#     if st.button("Classify Image"):
#         prediction = predict_image(image, model, transform)
#         st.write(f"Prediction: {prediction}")

# else:
#     st.write("Please upload an image to continue.")

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# Load the trained model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # For binary classification
model.load_state_dict(torch.load('madhubani_classifier.pth'))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
])

# Define the function to make predictions
def predict_image(image, model, transform):
    # Convert image to RGB if it has an alpha channel
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
    return "Madhubani" if predicted.item() == 0 else "Not Madhubani"

# Streamlit app layout
st.title("Madhubani Painting Classifier")
st.write("Upload an image of a painting to check if it's Madhubani style or not.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Add a button for prediction
    if st.button("Classify Image"):
        prediction = predict_image(image, model, transform)
        st.write(f"Prediction: {prediction}")

else:
    st.write("Please upload an image to continue.")