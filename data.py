import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import pickle
import os


class Data:

    def preprocess(self, img, image_shape):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])
        return transforms(img).unsqueeze(0)

    def extract_features(self, X, content_layers, style_layers):
        contents = []
        styles = []
        for i in range(len(self.net)):
            X = self.net[i](X)
            if i in self.style_layers:
                styles.append(X)
            if i in content_layers:
                contents.append(X)
        return contents, styles

    def get_styles(self, image_shape, device, style_img):
        style_X = self.preprocess(style_img, image_shape).to(device)
        _, styles_Y = self.extract_features(style_X, self.content_layers, self.style_layers)
        return style_X, styles_Y

    def gram(self, X):
        num_channels, n = X.shape[1], X.numel() // X.shape[1]
        X = X.reshape((num_channels, n))
        return torch.matmul(X, X.T) / (num_channels * n)

    def get_pattern(self):
        my_style = []
        for filename in os.listdir(self.path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(
                    ".bmp"):  # Add other extensions as needed
                img_path = os.path.join(self.path, filename)
                style_img = d2l.Image.open(img_path)
                _, styles_Y = self.get_styles(self.image_shape, self.device, style_img)
                styles_Y_gram = [self.gram(Y) for Y in styles_Y]
                my_style.append(styles_Y_gram)

        with open(self.path + "/style", "wb") as fp:
            pickle.dump(my_style, fp)

    def __init__(self, path):
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406])
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225])
        self.path = path

        self.pretrained_net = torchvision.models.vgg19(pretrained=True)

        self.style_layers, self.content_layers = [0, 5, 10, 19, 28], [25]

        net = nn.Sequential(*[self.pretrained_net.features[i] for i in
                              range(max(self.content_layers + self.style_layers) + 1)])

        self.device, self.image_shape = d2l.try_gpu(), (300, 450)
        self.net = net.to(self.device)
        self.net.eval()


data = Data("../yh")
data.get_pattern()
