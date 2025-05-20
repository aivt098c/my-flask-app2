from fastapi import FastAPI
import gradio as gr
from PIL import Image
import torch
import numpy as np
import io
import requests
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import uvicorn

app = FastAPI()

torch.set_float32_matmul_precision("high")
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda" if torch.cuda.is_available() else "cpu")

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_img(path_or_array):
    if isinstance(path_or_array, np.ndarray):
        return Image.fromarray(path_or_array.astype("uint8"))
    elif isinstance(path_or_array, str):
        if path_or_array.startswith("http"):
            response = requests.get(path_or_array)
            return Image.open(io.BytesIO(response.content))
        else:
            return Image.open(path_or_array)
    elif hasattr(path_or_array, "read"):
        return Image.open(path_or_array)
    else:
        raise ValueError(f"Unsupported input type: {type(path_or_array)}")

def process(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

def fn_uploaded(image):
    im = load_img(image)
    im = im.convert("RGB")
    original = im.copy()
    processed = process(im)
    return processed, original

interface = gr.Interface(
    fn_uploaded,
    inputs=gr.Image(label="Upload an Image"),
    outputs=[
        gr.Image(label="Processed Image", type="pil", format="png"),
        gr.Image(label="Original Image", type="pil")
    ],
    flagging_mode="never"
)

# ✅ Mount Gradio interface at /remove_background
app = gr.mount_gradio_app(app, interface, path="/remove_background")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
