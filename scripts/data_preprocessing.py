import os
import glob
from PIL import Image


from google.cloud import vision
from google.oauth2 import service_account

from dotenv import load_dotenv
load_dotenv()

CREDENTIALS_PATH = os.getenv("CREDENTIALS_PATH")
credentials = service_account.Credentials.from_service_account_file(
    CREDENTIALS_PATH)
client = vision.ImageAnnotatorClient(credentials=credentials)


RAW_DIR = 'data/raw/'
PROCESSED_DIR = 'data/processed/'


def detect_faces_with_api(path):
    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations
    if not faces:
        return None
    
    vertices = [
        vertex for vertex in faces[0].fd_bounding_poly.vertices
    ]

    return vertices


def preprocessing(raw_path, processed_path):
    vertices = detect_faces_with_api(raw_path)
    if not vertices:
        return

    image = Image.open(raw_path)
    face = image.crop((vertices[0].x, vertices[1].y,
                      vertices[2].x, vertices[3].y))

    face = face.resize((224, 224))
    face.save(processed_path)


if __name__ == '__main__':
    raw_paths = glob.glob(os.path.join(
        RAW_DIR, '**', '*.*'), recursive=True)

    for path in raw_paths:
        filename = os.path.basename(path)
        folder = os.path.basename(os.path.dirname(path))

        processed_folder = os.path.join(PROCESSED_DIR, folder)
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)

        processed_path = os.path.join(processed_folder, filename)

        preprocessing(path, processed_path)
