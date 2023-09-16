import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer
from utils.torch_utils import select_device, TracedModel
from utils.plots import plot_one_box
# from google.colab.patches import cv2_imshow




from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os
import cv2
import fitz


model_weights = './runs/train/yolov7-custom/weights/best.pt'

# Load the model
model_chart_type = load_model("/workspaces/chart_detection/model_weights/chart_classification/keras_model.h5", compile=False)

# Load the labels
class_names = open("/workspaces/chart_detection/model_weights/chart_classification/labels.txt", "r").readlines()

# Load the model
is_chart_model = load_model("/workspaces/chart_detection/model_weights/is_chart/keras_model.h5", compile=False)

# Load the labels
is_chart_class_names = open("/workspaces/chart_detection/model_weights/is_chart/labels.txt", "r").readlines()


def detect_objects(input_image, model_weights, img_size=640, conf_thres=0.25, iou_thres=0.45, device='cpu', save_img=False):
    charts=[]
    # # Initialize
    # device = select_device(device)
    # half = device.type != 'cpu'

    # # Load model
    # model = attempt_load(model_weights, map_location=device)
    # stride = int(model.stride.max())  # model stride
    # img_size = check_img_size(img_size, s=stride)  # check img_size

    # # Trace the model if needed
    # model = TracedModel(model, device, img_size)

    # if half:
    #     model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(input_image, img_size=img_size, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Save results (image with detections)
                if save_img:
                    for *xyxy, conf, class_id in reversed(det):
                      if conf < 0.75:
                        continue
                      label = f'{names[int(class_id)]} {conf:.2f}'
                      plot_one_box(xyxy, im0, label=label, color=colors[int(class_id)], line_thickness=1)
                      x0,y0,x1,y1 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                      charts.append((x0,y0,x1,y1))

    return im0,charts

device='cpu'
img_size=640


# Initialize
device = select_device(device)
half = device.type != 'cpu'

# Load model
model = attempt_load(model_weights, map_location=device)
stride = int(model.stride.max())  # model stride
img_size = check_img_size(img_size, s=stride)  # check img_size


# Trace the model if needed
model = TracedModel(model, device, img_size)

if half:
    model.half()  # to FP16



def processimages(image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    return data

def chart_classif(image_path):
  data =processimages(image_path)
  # Predicts the model
  prediction = model_chart_type.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]

  # # Print prediction and confidence score
  # print("Class:", class_name[2:], end="")
  # print("Confidence Score:", confidence_score)

  return class_name[2:]

def is_chart(image_path):
  data =processimages(image_path)
  # Predicts the model
  prediction = is_chart_model.predict(data)
  index = np.argmax(prediction)
  class_name = is_chart_class_names[index]
  confidence_score = prediction[0][index]

  # Print prediction and confidence score
  # print("Class:", int(class_name[2:]), end="")
  # print("Confidence Score:", confidence_score)

  return True if int(class_name[2:]) == 1 and confidence_score>0.99 else False



def detect_charts(pdf_file):
  folder_path = os.path.dirname(pdf_file)
  extracted_charts=[]
  pdf_document = fitz.open(pdf_file)
  for page_number in range(len(pdf_document)):
    page = pdf_document[page_number]
    pix = page.get_pixmap()
    pix.save("page-%i.png" % page.number)

    result_image ,bbox_list= detect_objects(f"page-{page.number}.png", model_weights, img_size=640, conf_thres=0.25, iou_thres=0.45, save_img=True)

    temp_path=f"page-{page.number}.png"
    img = cv2.imread(temp_path)

    for img_index,bbox in enumerate(bbox_list):

      cropped_img= img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
      path=os.path.join(folder_path,f"image_{img_index}_{page_number}.png")
      cv2.imwrite(path,cropped_img)
      if is_chart(path):
        chart_type=chart_classif(path)
        chart_list={"page_no":page_number,"chart_type":chart_type[:-1],"chart_path":path,"bbox":bbox}
        extracted_charts.append(chart_list)
      else:
        os.remove(path)
    os.remove(temp_path)
  return extracted_charts
      # chart_classif(cropped_img)

result=detect_charts("/workspaces/chart_detection/Wolters-Kluwer-2022-Annual Report-1 (1) (1).pdf")
print(result)