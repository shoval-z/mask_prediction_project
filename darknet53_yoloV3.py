from torch import nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from dataset import mask_dataset
import numpy as np
from utils import save_model
from utils import calc_iou
import warnings


warnings.filterwarnings("ignore", category=UserWarning)  # to ignore the .to(dtype=torch.uint8) warning message


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.darknet53 = self.darknet_53()
        self.bbox = nn.Sequential(
            nn.Conv2d(1024, 5, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(5),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid()
        )

    def darknet_53(self):
        return nn.Sequential(
            convolutional_layer(3, 32, kernel_size=3, padding=1, stride=1),
            convolutional_layer(32, 64, kernel_size=3, padding=1, stride=2),
            nn.Sequential(*[residual_block(in_channels=64, out_channels=32) for _ in range(1)]),  # residual
            convolutional_layer(64, 128, kernel_size=3, padding=1, stride=2),
            nn.Sequential(*[residual_block(in_channels=128, out_channels=64) for _ in range(2)]),  # residual
            convolutional_layer(128, 256, kernel_size=3, padding=1, stride=2),
            nn.Sequential(*[residual_block(in_channels=256, out_channels=128) for _ in range(8)]),  # residual
            convolutional_layer(256, 512, kernel_size=3, padding=1, stride=2),
            nn.Sequential(*[residual_block(in_channels=512, out_channels=256) for _ in range(8)]),  # residual
            convolutional_layer(512, 1024, kernel_size=3, padding=1, stride=2),
            nn.Sequential(*[residual_block(in_channels=1024, out_channels=512) for _ in range(4)]),  # residual
        )

    def forward(self, image: torch.Tensor):
        features = self.darknet53(image)  # [batch size, 1024, 10, 10]
        out = self.bbox(features).flatten(start_dim=1)
        classifier = out[:, 0]
        bbox = out[:, 1:]
        return classifier, bbox


def convolutional_layer(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(residual_block, self).__init__()
        self.conv = nn.Sequential(
            convolutional_layer(in_channels, out_channels, kernel_size=1, padding=0, stride=1),
            convolutional_layer(out_channels, in_channels, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, x):
        return x + self.conv(x)


def eval_only(model, device, test_loader, test_dataset, acc_loss_weight=1.):
    print('strat eval')
    pred_lst, real_lst = [], []
    model.eval()
    total = 0
    sum_loss = 0
    sum_iou = 0
    accuracy = 0
    loss_bb = torch.nn.L1Loss()
    loss_class = torch.nn.BCELoss()
    with torch.no_grad():
        for idx, (x, y_bb, y_class,origin_size) in enumerate(test_loader):
            origin_size = origin_size.squeeze(1).to(device)

            batch = y_class.shape[0]
            x = x.to(device).float()
            y_class = y_class.to(device)
            y_class = (y_class == 2).squeeze(1).float()
            y_bb = y_bb.to(device).float()
            out_class, out_bb = model(x)

            out_bb = torch.mul(out_bb, origin_size)
            y_bb = torch.mul(y_bb.squeeze(1), origin_size)

            loss = loss_bb(out_bb, y_bb.squeeze(1))
            loss += acc_loss_weight * (loss_class(out_class, y_class))
            total += batch
            sum_loss += loss.item()

            pred = (out_class > 0.5).float()
            accuracy += pred.eq(y_class).sum().item()

            tmp_iou = [calc_iou(det_b, true_b) for det_b, true_b in
                       zip(out_bb.cpu().detach().numpy(), y_bb.squeeze(1).cpu().detach().numpy())]
            sum_iou += np.sum(tmp_iou)

        test_loss = sum_loss / total
        test_acc = accuracy / total
        test_iou = sum_iou / total
        print("Test_iou %.3f,test_accuracy %.3f,test_loss %.3f  " % (
            test_iou, test_acc, test_loss))
    # return sum_iou / total, accuracy / total, sum_loss / total, real_lst, pred_lst


def train_and_eval_single_epoch(model, device, optimizer, train_loader, train_dataset, test_loader, test_dataset, epoch,
                                acc_loss_weight=1., clipping=False, norm=20):
    print('start training')
    loss_bb = torch.nn.L1Loss()
    loss_class = torch.nn.BCELoss()
    model.train()
    total = 0
    sum_loss = 0
    sum_iou = 0
    accuracy = 0
    for idx, (x, y_bb, y_class, origin_size) in enumerate(train_loader):  # x = [batch_size, RGB, 300, 300]
        origin_size = origin_size.squeeze(1).to(device)

        batch = y_class.shape[0]
        x = x.to(device).float()
        y_class = y_class.to(device)
        y_class = (y_class == 2).squeeze(1).float()
        y_bb = y_bb.to(device).float()
        out_class, out_bb = model(x)

        out_bb = torch.mul(out_bb, origin_size)
        y_bb = torch.mul(y_bb.squeeze(1), origin_size)

        loss = loss_bb(out_bb, y_bb.squeeze(1))
        loss += acc_loss_weight * (loss_class(out_class, y_class))
        optimizer.zero_grad()
        loss.backward()
        # printing gradients norms
        max_norm = 0
        # for name, param in model.named_parameters():
        #     norm = param.grad.norm(2)
        #     # print(name, norm)
        #     if norm > max_norm:
        #         max_norm = norm
        # print(f'MAX NORM = {max_norm}')
        if clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=norm, norm_type=2)
        optimizer.step()
        total += batch
        sum_loss += loss.item()


        pred = (out_class > 0.5).float()
        accuracy += pred.eq(y_class).sum().item()

        tmp_iou = [calc_iou(det_b, true_b) for det_b, true_b in
                   zip(out_bb.cpu().detach().numpy(), y_bb.squeeze(1).cpu().detach().numpy())]
        sum_iou += np.sum(tmp_iou)


    train_loss = sum_loss / total
    train_acc = accuracy / total
    train_iou = sum_iou / total
    print("for epoch: %f \t train_iou %.3f,train_accuracy %.3f,train_loss %.3f  " % (
        epoch, train_iou, train_acc, train_loss))

    ## eval
    print('strat eval')
    model.eval()
    total = 0
    sum_loss = 0
    sum_iou = 0
    accuracy = 0
    loss_bb = torch.nn.L1Loss()
    loss_class = torch.nn.BCELoss()
    with torch.no_grad():
        for idx, (x, y_bb, y_class, origin_size) in enumerate(test_loader):
            origin_size = origin_size.squeeze(1).to(device)

            batch = y_class.shape[0]
            x = x.to(device).float()
            y_class = y_class.to(device)
            y_class = (y_class == 2).squeeze(1).float()
            y_bb = y_bb.to(device).float()
            out_class, out_bb = model(x)

            out_bb = torch.mul(out_bb, origin_size)
            y_bb = torch.mul(y_bb.squeeze(1), origin_size)

            loss = loss_bb(out_bb, y_bb.squeeze(1))
            loss += acc_loss_weight * (loss_class(out_class, y_class))
            total += batch
            sum_loss += loss.item()

            pred = (out_class > 0.5).float()
            accuracy += pred.eq(y_class).sum().item()

            tmp_iou = [calc_iou(det_b, true_b) for det_b, true_b in
                       zip(out_bb.cpu().detach().numpy(), y_bb.squeeze(1).cpu().detach().numpy())]
            sum_iou += np.sum(tmp_iou)

        test_loss = sum_loss / total
        test_acc = accuracy / total
        test_iou = sum_iou / total
        print("for epoch: %f \t test_iou %.3f,test_accuracy %.3f,test_loss %.3f  " % (
            epoch, test_iou, test_acc, test_loss))


    return train_iou, train_acc, train_loss, test_iou, test_acc, test_loss, model


def main():
    training = False
    # Learning parameters
    batch_size = 64
    lr = 0.001
    acc_loss_weight = 0.5
    norm=10
    workers = 4
    clipping = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if training:
        model = Darknet53().to(device)
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=lr)

        train_dataset = mask_dataset(dataset='train', path=f'/home/student/train')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=workers,
                                                   pin_memory=True)
        test_dataset = mask_dataset(dataset='test', path=f'/home/student/test')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=workers,
                                                  pin_memory=True)
        train_loss_list, train_iou_list, train_acc_list = list(), list(), list()
        test_loss_list, test_iou_list, test_acc_list = list(), list(), list()
        for epoch in range(50):
            train_iou, train_acc, train_loss, test_iou, test_acc, test_loss, new_model = train_and_eval_single_epoch(model,
                                                                                                                     device,
                                                                                                                     optimizer,
                                                                                                                     train_loader,
                                                                                                                     train_dataset,
                                                                                                                     test_loader,
                                                                                                                     test_dataset,
                                                                                                                     epoch,
                                                                                                                     acc_loss_weight,
                                                                                                                     clipping=clipping,
                                                                                                                     norm=norm)
            train_iou_list.append(train_iou)
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            test_iou_list.append(test_iou)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)

            if test_iou > 0.615 and test_acc > 0.75:
                print('saving model')
                save_model(f'{epoch}_darknet53_{batch_size}_{lr}_{acc_loss_weight}_clipping={clipping}', new_model)

            print('train_iou=', train_iou_list)
            print('train_accuracy=', train_acc_list)
            print('train_loss=', train_loss_list)
            print('test_iou=', test_iou_list)
            print('test_accuracy=', test_acc_list)
            print('test_loss=', test_loss_list)

    else: #eval only
        model = torch.load('27_darknet53_64_0.001_0.5_clipping=True.pth.tar')
        model.to(device)
        test_dataset = mask_dataset(dataset='test', path=f'/home/student/test')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=workers,
                                                  pin_memory=True)

        eval_only(model, device, test_loader, test_dataset, acc_loss_weight=acc_loss_weight)


if __name__ == '__main__':
    main()
