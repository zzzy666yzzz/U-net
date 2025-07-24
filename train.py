
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from data import *
from net import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = 'VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
save_path = 'train_image'

if __name__ == '__main__':
    data_loader = DataLoader(MydataSet(data_path), batch_size=5, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters(),lr=0.001)
    loss_fun = nn.CrossEntropyLoss()

    epoch = 1
    train_loss = 0
    while epoch < 10:
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)
            segment_image = segment_image.squeeze(1).long()

            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()


        print(f'{epoch}--train_loss===>>{train_loss.item()}')
        torch.save(net.state_dict(), weight_path)

            # _image = image[0]
            # _segment_image = segment_image[0]
            # _out_image = out_image[0].argmax(dim = 0, keepdims=True).float()
            #
            # _image_rgb = _image
            # _segment_image_rgb = convert_to_rgb(_segment_image)
            # _out_image_rgb = convert_to_rgb(_out_image)
            #
            # img = torch.stack([_image, _segment_image.unsqueeze(0), _out_image], dim=0)
            # save_image(img, f'{save_path}/{epoch}_{i}.png')

        epoch += 1
