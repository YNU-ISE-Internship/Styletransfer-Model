import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import pickle
from tqdm import tqdm
import os


class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


class Loss:

    def content_loss(self, Y_hat, Y):
        # 我们从动态计算梯度的树中分离目标：
        # 这是一个规定的值，而不是一个变量。
        return torch.square(Y_hat - Y.detach()).mean()

    def gram(self, X):
        num_channels, n = X.shape[1], X.numel() // X.shape[1]
        X = X.reshape((num_channels, n))
        return torch.matmul(X, X.T) / (num_channels * n)

    def style_loss(self, Y_hat, gram_Y):
        return torch.square(self.gram(Y_hat) - gram_Y.detach()).mean()

    def tv_loss(self, Y_hat):
        return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                      torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

    def compute_loss(self, X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
        # 分别计算内容损失、风格损失和全变分损失
        contents_l = [self.content_loss(Y_hat, Y) * self.content_weight for Y_hat, Y in zip(
            contents_Y_hat, contents_Y)]
        styles_l = [self.style_loss(Y_hat, Y) * self.style_weight for Y_hat, Y in zip(
            styles_Y_hat, styles_Y_gram)]
        tv_l = self.tv_loss(X) * self.tv_weight
        # 对所有损失求和
        l = sum(10 * styles_l + contents_l + [tv_l])
        return contents_l, styles_l, tv_l, l

    def __init__(self, content_weight=1, style_weight=1e3, tv_weight=10):
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight


class Generate:

    def preprocess(self, img, image_shape):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])
        return transforms(img).unsqueeze(0)

    def postprocess(self, img):
        img = img[0].to(self.rgb_std.device)
        img = torch.clamp(img.permute(1, 2, 0) * self.rgb_std + self.rgb_mean, 0, 1)
        return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

    def extract_features(self, X, content_layers, style_layers):
        contents = []
        styles = []
        for i in range(len(self.net)):
            X = self.net[i](X)
            if i in style_layers:
                styles.append(X)
            if i in content_layers:
                contents.append(X)
        return contents, styles

    def get_contents(self, image_shape, device):
        content_X = self.preprocess(self.content_img, image_shape).to(device)
        contents_Y, _ = self.extract_features(content_X, self.content_layers, self.style_layers)
        return content_X, contents_Y

    def gram(self, X):
        num_channels, n = X.shape[1], X.numel() // X.shape[1]
        X = X.reshape((num_channels, n))
        return torch.matmul(X, X.T) / (num_channels * n)

    def get_inits(self, X, device, lr):
        gen_img = SynthesizedImage(X.shape).to(device)
        gen_img.weight.data.copy_(X.data)
        trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
        return gen_img(), trainer

    def train(self, lr=0.3, num_epochs=500, lr_decay_epoch=50):
        X, contents_Y = self.get_contents(self.image_shape, self.device)
        X, trainer = self.get_inits(X, self.device, lr)
        scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
        # animator = d2l.Animator(xlabel='epoch', ylabel='loss',
        #                         xlim=[10, num_epochs],
        #                         legend=['content', 'style', 'TV'],
        #                         ncols=2, figsize=(7, 2.5))
        loss = Loss(content_weight=1, style_weight=1e3, tv_weight=10)
        for epoch in tqdm(range(num_epochs)):
            trainer.zero_grad()
            contents_Y_hat, styles_Y_hat = self.extract_features(
                X, self.content_layers, self.style_layers)
            contents_l, styles_l, tv_l, l = loss.compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, self.style_Y)
            l.backward()
            trainer.step()
            scheduler.step()
            # if (epoch + 1) % 10 == 0:
            #     animator.axes[1].imshow(self.postprocess(X))
            #     animator.add(epoch + 1, [float(sum(contents_l)),
            #                              float(sum(styles_l)), float(tv_l)])
        return X

    def __init__(self, model_type, path_style, path_image, image_shape):
        # self.image_shape = (300, 450)
        self.image_shape = image_shape
        self.content_img = d2l.Image.open(path_image)
        with open(path_style + "/style_" + model_type, "rb") as fp:
            Y = pickle.load(fp)
        self.style_Y = [0.1 * y1 + 0.9 * y2 for y1, y2 in zip(Y[1], Y[5])]

        self.device = torch.device("cuda")

        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406])
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225])

        if model_type == "standard":
            self.style_layers, self.content_layers = [0, 5, 10, 19, 28], [25]
            self.pretrained_net = torchvision.models.vgg19(pretrained=True)
            net = nn.Sequential(*[self.pretrained_net.features[i] for i in
                                range(max(self.content_layers + self.style_layers) + 1)])
        elif model_type == "light":
            self.style_layers, self.content_layers = [1, 2, 4, 7, 14], [4]
            self.pretrained_net = torchvision.models.mobilenet_v3_large(pretrained=True)
            net = nn.Sequential(*[self.pretrained_net.features[i] for i in
                                  range(max(self.content_layers + self.style_layers) + 1)])
        self.net = net.to(self.device)
        self.net.eval()


a = Generate("light", "../yh", '../img/GOEY7(QY}M(H(PPY[]X3`82_tmb.jpg', (480, 600))
output = a.train(num_epochs=2000, lr_decay_epoch=250)
output_image = a.postprocess(output)  # Convert the tensor to an image
output_image_path = '../yh/generated_image.jpg'  # Specify the local path to save the image
output_image.save(output_image_path)  # Save the image