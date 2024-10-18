from scripts.inference import *
from models import model_architecture
import torch
from PIL import Image
import gradio as gr

model = torch.load('output/model/mode1.pth')


def face_recognition(path):
    image = Image.open(path)
    image, faces = detect_faces(image)
    labels = []
    for face in faces:
        labels.append(indentify_faces(model, face))
    text = ''
    for i in range(len(labels)):
        text += f'{i}. {labels[i]}\n'
    return image, text


interface = gr.Interface(fn=face_recognition,
                         inputs=gr.Image(type="filepath", height=512),
                         outputs=[gr.Image(height=512), gr.Textbox()],
                         examples=[])
interface.launch(share=True)