import os
import argparse
import numpy as np
from scipy import stats
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as standard_transforms
from .models import build_model


#CONSTANTS
w = 320*2
h = 240*2
threshold = 0.85
#if use gpu uncomment flowing lines
device = torch.device('cuda')
#################################

# Create P2P-Net model and load it's weights
def create_model():
  args = argparse.Namespace(backbone='vgg16_bn', gpu_id=0, line=2, row=2, weight_path='P2PNet/weights/SHTechA.pth')

  #if use gpu uncomment flowing lines
  os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

  # build model
  model = build_model(args)

  #if use gpu uncomment flowing lines
  model.to(device)

  # load trained model
  if args.weight_path is not None:
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

  # convert to eval mode
  model.eval()

  # create the pre-processing transform
  transform = standard_transforms.Compose([
    standard_transforms.ToTensor(), 
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  return model, transform

# Predict location of heads
def predict(img_path, model, transform):

  # read and resize image
  img_raw = Image.open(img_path).convert('RGB')
  img_raw = img_raw.resize((w, h), Image.ANTIALIAS)

  # pre-processe image
  img = transform(img_raw)
  samples = torch.Tensor(img).unsqueeze(0)

  #if use gpu uncomment flowing lines
  samples = samples.to(device)
    
  # run inference
  outputs = model(samples)

  # extract scores and locations of output points 
  outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
  outputs_points = outputs['pred_points'][0]

  # filter the predictions by a threshold to find suitable points
  points = outputs_points[outputs_scores > threshold].detach().cpu().numpy()

  return points

# Number of heads
def count(points):
  return points.shape[0]

# Draw points on image
def draw_points(img_path, points):
  # read and resize image
  img_raw = Image.open(img_path).convert('RGB')
  img_raw = img_raw.resize((w, h), Image.ANTIALIAS)
  size = 5
  # draw points in image
  img_raw = np.array(img_raw)
  output = img_raw.copy()
  for p in points:
    output = cv2.circle(output, (int(p[0]), int(p[1])), size, (255, 0, 0), -1)

  return output

# Create density map
def density_map(points):
  # calculate density PDF
  points=np.array(points).T
  points[[0, 1]] = points[[1, 0]]
  kernel = stats.gaussian_kde(points)
  # calculate density in image
  X, Y = np.mgrid[0:h, 0:w]
  positions = np.vstack([X.ravel(), Y.ravel()])
  density_map = np.reshape(kernel(positions).T, X.shape)

  return density_map
