from gevent import pywsgi
import cv2
from PIL import Image
import os
import numpy as np
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import json
from flask import Flask, Response, request, render_template, make_response, jsonify
from fastsam import FastSAM, FastSAMPrompt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(BASE_DIR)

app = Flask(__name__)

model_type = "vit_t"
sam_checkpoint = "../weights/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("device:", device)
print("modelsam加载成功")
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
predictor = SamPredictor(mobile_sam)

model = FastSAM(r'../weights/FastSAM_X.pt')
print("fastsam加载成功")


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        else:
            return super(MyEncoder, self).default(obj)


@app.route('/upload_img', methods=['POST'])
def upload_img():
    image = request.files["img"]
    image_pil = Image.open(image.stream)
    img_bgr = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)
    predictor.set_image(img_bgr)
    return "上传成功"


@app.route('/point_predict', methods=['POST'])
def point_predict():
    points = np.array(eval(request.form["points"]), dtype=np.float32)
    labels = np.array(eval(request.form["labels"]), dtype=np.int32)

    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        # mask_input=self.mask_input[None, :, :] if self.mask_input is not None else None,
        multimask_output=False,
    )
    # mask = masks[np.argmax(scores), :, :] 
    mask = masks[0]

    # ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    new_contours = []
    for c in contours:
        l = len(c)
        new_contours.append(c.reshape(l, -1))

    result = {
        "contours": new_contours
    }

    return json.dumps(result, cls=MyEncoder)


@app.route('/box_predict', methods=['POST'])
def box_predict():
    box = np.array(eval(request.form["box"]), dtype=np.float32)

    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=False,
    )
    # mask = masks[np.argmax(scores), :, :] 
    mask = masks[0]

    # ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    new_contours = []
    for c in contours:
        l = len(c)
        new_contours.append(c.reshape(l, -1))

    result = {
        "contours": new_contours
    }

    return json.dumps(result, cls=MyEncoder)


@app.route('/auto_predict', methods=['POST'])
def auto_predict():
    DEVICE = 'cpu'
    image = request.files["img"]
    image_pil = Image.open(image.stream)
    IMAGE_PATH = "images/src.jpg"
    image_rgb = image_pil.convert('RGB')
    image_rgb.save(IMAGE_PATH)
    everything_results = model(
        IMAGE_PATH,
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)
    # everything prompt
    anns = prompt_process.everything_prompt()
    anns = np.uint8(anns.numpy() * 255)
    # ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    res = []
    for ann in anns:
        contours, _ = cv2.findContours(ann, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_contours = []
        for c in contours:
            l = len(c)
            new_contours.append(c.reshape(l, -1))
        res.extend(new_contours)

    result = {
        "contours": res
    }

    return json.dumps(result, cls=MyEncoder)


if __name__ == '__main__':
    # app.run(debug=False, port=8765)
    # app.run(debug=True, port=8765)
    server = pywsgi.WSGIServer(('0.0.0.0', 8765), app)
    server.serve_forever()
