from PIL import Image
####等比缩放
#Unet要求输入的图像具有相同的尺寸
def keep_image_size_open(path,size = (256,256),mode = 'RGB'):
    img = Image.open(path).convert(mode)#创建单通道模式的图像
    temp = max(img.size)
    mask = Image.new(mode,(temp,temp),(0,0,0) if mode == 'RGB' else 0)
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask