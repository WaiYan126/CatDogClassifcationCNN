import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import CNN


@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(
        torch.load(
            "cat_dog_cnn.pth",
            map_location=torch.device("cpu")
        )
    )
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱"
)

st.title("🐱🐶 Cat vs Dog Classifier")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    x = transform(image)
    x = x.unsqueeze(0)

    with torch.no_grad():

        logit = model(x)

        probability = torch.sigmoid(logit).item()

    if probability >= 0.5:
        prediction = "Dog"
        confidence = probability
    else:
        prediction = "Cat"
        confidence = 1 - probability

    st.subheader("Prediction")

    st.success(
        f"{prediction} ({confidence:.2%} confidence)"
    )

    st.write("Probabilities")

    st.progress(float(probability))

    st.write(f"Dog: {probability:.2%}")
    st.write(f"Cat: {(1 - probability):.2%}")