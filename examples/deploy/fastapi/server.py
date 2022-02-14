import cv2
import io
import uvicorn
import numpy as np

import megengine as mge
import megengine.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException

from model import LeNet

checkpoint_path = "output/checkpoint.pkl"
checkpoint = mge.load(checkpoint_path)
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]

model = LeNet()
model.load_state_dict(state_dict)

app = FastAPI(title="Deploying a MegEngine LeNet Model with FastAPI")


@app.get("/")
def home():
    return "The server is woking~ Now head over to http://localhost:8000/docs"


@app.post("/predict")
async def prediction(file: UploadFile = File(...)):

    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(
            status_code=415, detail="Unsupported file provided.")
    if file.content_type.startswith('image/') is False:
        raise HTTPException(
            status_code=400, detail="File is not an image."
        )

    contents = await file.read()
    image_stream = io.BytesIO(contents)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()))
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32))
    image = np.array(255 - image)
    image = mge.Tensor(image).reshape((1, 1, 32, 32))

    logit = model(image)
    label = F.argmax(logit).item()

    return {
        "filename": file.filename,
        "contenttype": file.content_type,
        "likely_class": label,
    }

uvicorn.run(app, host="127.0.0.1", port=8000)
