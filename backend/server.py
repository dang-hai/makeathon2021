import io
import cv2
import base64
from flask import request, send_file
from flask import Flask
from flask import jsonify
from flask_cors import CORS
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np
from pathlib import Path

import os
import datetime

import torch
from torch import nn

import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

class CubiCasa(torch.utils.data.Dataset):
	def __init__(self, data_root, transform=None):
		super(CubiCasa, self).__init__()
		self.data = [str(Path(data_root, f.name)) for f in Path(data_root).iterdir()]
		self.transform = transform

	def __getitem__(self, idx):
		img = cv2.imread(self.data[idx], 0)

		if self.transform:
			img = self.transform(img)
			
		return img, 0
	
	def __len__(self):
		return len(self.data)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1), # b, 8, 2, 2
            nn.Flatten(1),
            nn.Linear(3528, 128),
            nn.Linear(128, 5),
            nn.Softmax(1)
        )


    def forward(self, x):
        x = self.encoder(x)
        return x

drawing_ckpts = torch.load('../drawing_trained/sim_autoencoder.pth')
drawingModel = Classifier()
drawingModel.load_state_dict(drawing_ckpts)
drawingModel.eval()

target_size = 256
num_epochs = 1000
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])	

dataset = CubiCasa('../data/transformed', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model_path = '../trained/20211016_13:2442/sim_autoencoder.pth'

ckpts = torch.load(model_path)
model = autoencoder()
model.load_state_dict(ckpts)
model.eval()

X = np.vstack([d.numpy() for d, _ in dataset])
X = torch.from_numpy(X)
X = X.unsqueeze(1)

print("Fitting NN")
with torch.no_grad():
	model.eval()
	enc = model.encoder(X).cpu().reshape(X.shape[0], -1)
	# print(enc.shape)
	knn = NearestNeighbors()
	knn.fit(enc)

print("NN model fit finished.")

def preprocess(img, target_size=256):
	w, h = img.shape
	dst = img
	if w > target_size or h > target_size:
		## (1) Convert to gray, and threshold
		# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		threshed = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

		## (2) Morph-op to remove noise
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
		morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

		## (3) Find the max-area contour
		cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		cnt = sorted(cnts, key=cv2.contourArea)[-1]

		## (4) Crop and save it
		x,y,w,h = cv2.boundingRect(cnt)
		dst = img[y:y+h, x:x+w]

		max_size = np.max(dst.shape)
		if max_size > target_size:
			imgW, imgH = dst.shape
			r = target_size/max_size
			dim = (target_size if imgW > imgH else int(r*imgW), target_size if imgH > imgW else int(r*imgH))
			dst = cv2.resize(dst, dim)

		w, h = dst.shape
		padding_width = (target_size - w)
		padding_left = int(padding_width/2)
		padding_right = padding_width - padding_left
		padding_height = (target_size - h)
		padding_top = int(padding_height/2)
		padding_bot = padding_height - padding_top
		dst = cv2.copyMakeBorder(dst, padding_left, padding_right, padding_top, padding_bot, cv2.BORDER_CONSTANT, value=[255, 255, 255])

	# Morph open to remove noise
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
	opening = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel, iterations=1)

	# Find contours and remove small noise
	cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	for c in cnts:
		area = cv2.contourArea(c)
		if area < 50:
			cv2.drawContours(opening, [c], -1, 0, -1)

	# Invert and apply slight Gaussian blur
	dst = 255 - opening
	dst = cv2.GaussianBlur(dst, (3,3), 0)

	return dst

print("Starting Server ...")
def create_app():
	app = Flask(__name__)
	CORS(app)

	@app.route("/")
	def helloWorld():
		return "Hello, cross-origin-world!"

	@app.route("/api/search", methods=["POST"])
	def search():
		filestr = request.files['image'].read()
		#convert string data to numpy array
		npimg = np.fromstring(filestr, np.uint8)

		# convert numpy array to image
		img = cv2.imdecode(npimg, cv2.COLOR_BGR2GRAY)

		img = preprocess(img)
		w, h = img.shape

		cv2.imwrite('test.png', img)

		# decode image
		img = torch.from_numpy(img).float().reshape(1, 1, w, h)

		with torch.no_grad():
			model.eval()
			enc = model.encoder(img)

		res = knn.kneighbors(enc.cpu().reshape(1, -1), 10)

		print(res)
		files = [Path(d).name for idx, d in enumerate(dataset.data) if idx in res[1]]

		return jsonify({"msg": "ok", "content": files})
	
	@app.route("/api/image", methods=["GET"])
	def image():
		image_file = Path(request.args.get('image')).name

		data_root = "../data/original"
		path = Path(data_root, image_file)

		return send_file(path, mimetype='image/png')

	@app.route("/api/drawing", methods=["POST"])
	def analyse_drawing():
		uri = request.json['img']
		encoded_data = uri.split(',')[1]

		base64_decoded = base64.b64decode(encoded_data)

		image = Image.open(io.BytesIO(base64_decoded))

		img = np.array(image)[:, :, 3]
		img = np.invert(img)
		cv2.imwrite('client.png', img)
		img = preprocess(img)
		cv2.imwrite('test.png', img)

		img = torch.from_numpy(img).float()
		pred = drawingModel(img.unsqueeze(0).unsqueeze(0)).argmax()

		data_dir = Path(f"../fp_map_drawing/{pred}")

		files = [p.name for p in data_dir.iterdir() if p.suffix == '.png']

		return jsonify({"msg": "ok", "content": files})

	return app

app = create_app()
app.run(debug=False)