from django.http import HttpResponse
# from django.shortcuts import render
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from joblib import load
from subprocess import run
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import cv2
import torch
from torchvision import models
from django.conf import settings


# model = load(".\savedmodels\my_yolov5_model.joblib")


# from . models import MyModel

    
# def upload_image(request):

def index(request):
    return render(request, 'index.html')


def analyze(request):
    if request.method == 'POST':
        uploaded_image = request.FILES['image_input']
        # uploaded_image.name = 'weed.jpg'
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        # image_path = fs.url(filename)
        static_image_path = 'static/' + uploaded_image.name  # Adjust the path accordingly

        # Save the uploaded image to the static path
        fs.save(static_image_path, uploaded_image)
            # Call detect.py with image path as argument
    # process = run(['python', '.\savedmodels\detect.py', '--weights', '.\\best.pt', '--source', image_path])
# Assuming you've loaded your pre-trained model into the 'model' variable
  # Adjust this import based on your actual model architecture

        # Assuming you have defined your model architecture
        # Replace the following line with the actual definition of your model
        model = models.resnet50(pretrained=False, num_classes=4)

        # Specify the path to your saved model file
        model_path = './savedmodels/mymodel.pth'  # Adjust the path accordingly

        # Load the model state dictionary
        model.load_state_dict(torch.load(model_path))

        # Set the model to evaluation mode
        model.eval()

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Assuming your model was trained with this input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define a dictionary to map class indices to class labels
        class_mapping = {
            0:'Maize',
            1: 'barnyard',
            2: 'foxtail',
            3: 'pigweed'
        }

        # Path to the folder containing test images
        # test_folder_path = 'test_set'

        # List all image files in the folder
        # image_files = [f for f in os.listdir(test_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        
        herbicide=""
        # Evaluate the model on the test images
        with torch.no_grad():
        #     for image_file in image_files:
        #         # Load and preprocess the image using OpenCV
        #         image_path = os.path.join(test_folder_path, image_file)
            # image = cv2.imread(image_path)
            image = cv2.imread(uploaded_image.name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = cv2.resize(image, (224, 224))  # Resize
            image = transforms.ToPILImage()(image)  # Convert to PIL Image
            input_tensor = test_transform(image).unsqueeze(0)  # Add batch dimension

                # Get the model prediction
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

                # Map the numerical prediction to the class label
            predicted_label = class_mapping[predicted.item()]
            
            
            
            if predicted_label=='barnyard':
                herbicide='A'
            if predicted_label=='foxtail':
                herbicide='B'
            if predicted_label=='pigweed':
                herbicide='C'

                # Display the image along with the predicted label
            # plt.imshow(image)
            # plt.title(f"Predicted Class: {predicted_label}")
            # plt.axis('off')
            # plt.show()
            params = { 'label': predicted_label, 'img_pth':uploaded_image.name, 'herb': herbicide}
            # return render(request, 'result.html', params)
            os.remove(os.path.join(settings.MEDIA_ROOT,filename))
            return render(request, 'result.html', params)
def delete(request):  
    # if request.method=='POST':
    #     os.remove(os.path.join(settings.MEDIA_ROOT,img_pth))        
    return render(request,'index.html')
            # return render(request,plt.show())
                # Update your model here
                # obj = MyModel.objects.create(image=image_path
            #     return redirect('success_page')
            # return render(request, 'upload_image.html')
                





