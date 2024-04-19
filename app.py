import torch.nn as nn
import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import streamlit as st  
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image , ImageOps

#streamlit code

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Pathogen Detection",
    page_icon = "🌿",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML
    
class ResNetTune(nn.Module):
    def __init__(self, model):
        super(ResNetTune, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return nn.functional.softmax(x, dim=1)
    

def load_model():
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs,5)
    model = torch.load("./pathogen_classification_model.pth" , map_location=torch.device('cpu'))
    model.eval()
    return model
with st.spinner('Model is being loaded..'):
    model =load_model()


def predict(img):
    
    SIZE = 224
    class_indices = {0: 'Bacteria', 1: 'Fungi', 2: 'Healthy', 3: 'Pests', 4: 'Virus'}
    transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Resize((SIZE, SIZE)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image = Image.open(img).convert("RGB")
    image = transform(np.array(image))
    image = image.view(1, 3, SIZE, SIZE)
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            image = image.cuda()

        out = model(image)
        predicted_class_index= out.argmax(1).item()
        predicted_class_name = class_indices[predicted_class_index]
        return predicted_class_name


with st.sidebar:
        st.image('plant-header.png')
        st.title("Plant Pathogen Detection")
        st.subheader("Accurate detection of pathogens present in the plants. This helps an user to easily detect the pathogens and identify it's remedy.")

st.write("""
         # PlantPath : Plant Pathogen Detection
         """
         )

file = st.file_uploader("Safeguarding Agricultural Health", type=["jpg", "png","jpeg"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = predict(file)
    remedies = {'Bacteria': 'Use copper-based fungicides or bactericides to control bacterial infections in plants.',
                'Fungi':'Apply fungicides containing active ingredients such as azoxystrobin or propiconazole to manage fungal infections in plants.',
                'Pests' : "Implement integrated pest management strategies including biological control, cultural practices, and targeted pesticide application to control pest infestations in crops.",
                'Virus' : "Practice strict sanitation measures and remove infected plants to prevent the spread of viral diseases in crops." }

    string = "Detected Pathogen : " + predictions
    if predictions == 'Healthy':
        st.balloons()
        st.sidebar.success(string)

    else: 
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(remedies[predictions])

    





         
