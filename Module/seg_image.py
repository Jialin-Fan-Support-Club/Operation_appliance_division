import os
import numpy as np
import cv2
import torch
from albumentations import (
    PadIfNeeded,
    Normalize,
    Compose)
import time
import math
from Module.parts_seg import CleanU_Net as parts_seg
from albumentations.pytorch.transforms import img_to_tensor
from Module.binary_seg_model import CleanU_Net as binary_seg

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files  # 获取当前路径下所有非目录子文件名称


def mask_overlay(image, mask, color=(0, 255, 0)):  # 将分割结果作为mask放入原图
    """
    Helper function to visualize mask on the top of the car
    """
    # mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8) * 255
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img


def illum(img):  # 对图像中的高亮部分进行模糊处理
    # img = cv2.imread("test2.jpg")
    # img = img[532:768, 0:512]
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_bw, 219, 255, 0)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # print(cnts.shape)
    # cnts = cnts[0]
    img_zero = np.zeros(img.shape, dtype=np.uint8)
    # img[thresh == 255] = 150
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        img_zero[y:y + h, x:x + w] = 255
    # cv2.imshow("mask", mask)
    mask = img_zero
    # cv2.imshow("mask", mask)
    result = cv2.illuminationChange(img, mask, alpha=0.4, beta=0.4)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    return result


def splitFrames(videoFileName, data_path):  # 视频地址，图像保存地址,把视频抽取为图像进行分割
    cap = cv2.VideoCapture(videoFileName)
    success, data = cap.read()
    i = 0
    timeF = int(round(cap.get(cv2.CAP_PROP_FPS)))  # 帧率，每秒多少帧
    j = 0
    print(success)
    while success:
        i = i + 1
        if (i % timeF == 0):  # 按一秒一帧提取图像
            j = j + 1
            # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            # plt.imshow(data)
            # plt.show()
            cv2.imwrite(data_path + "/" + str(j) + ".jpg", data)
            # save_image(data, r'C:\Users\罗峥嵘\Desktop\avi/', j)
            print('save image:', j)
        success, data = cap.read()
    cap.release()


def save_video(data_path, path_save, h, w, filename):
    fps = 1

    size = (w, h)  # 获取图片宽高度信息
    print(size)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    videoWrite = cv2.VideoWriter(path_save + "/" + filename+'.avi', fourcc, fps,  # 视频保存地址
                                 size)  # 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））
    # videoWrite = cv2.VideoWriter('0.mp4',fourcc,fps,(1920,1080))

    files = os.listdir(data_path)  # 图像地址
    out_num = len(files)
    for i in range(out_num):
        fileName = data_path + "/" + str(i).zfill(4) + '.png'  # 循环读取所有的图片,假设以数字顺序命名
        img = cv2.imread(fileName)
        videoWrite.write(img)  # 将图片写入所创建的视频对象


def main(seg, type, data_path, result_path, result_name):

    class_color = [[0, 0, 0], [0, 255, 0], [0, 255, 255], [125, 255, 12]]

    DEVICE = torch.device("cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    t1 = time.time()
    # ********
    # seg = "binary"  # 多类别分割，夹子为一类，铰链为一类，柄为一类

    if seg == "parts":
        model = parts_seg(in_channels=3, out_channels=4)
        model_path = r"model\model_2_TDSNet.pt"  # 多类别分割模型地址
    elif seg == "binary":  # 二元分割，背景为一类手术器械为一类
        model = binary_seg(in_channels=3, out_channels=2)
        model_path = r"model\model_0_TDSNet.pt"  # 二元分割模型地址
    else:
        model = r""
        model_path = r""

    state = torch.load(str(model_path), map_location='cpu')
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    # **********
    # type = "image"
    # path = r"C:\学习资料\大二下\佳林哥哥\images1"  # 要分割的图像或者视频或者nii所在的文件夹地址
    # path_save = r"C:\学习资料\大二下\佳林哥哥\test3"  # 要保存图像的地址
    names = file_name(data_path)  # path目录下所有文件名字
    factor = 2 ** 5
    # ********

    if type == "image":
        for i, name in enumerate(names):
            t1 = time.time()
            path_single = os.path.join(data_path, name)
            image = cv2.imdecode(np.fromfile(path_single, dtype=np.uint8), -1)

            h, w, channel = image.shape


            h = math.ceil(h / factor) * factor // 2  # 向上取整，由于模型需要下采样5次图像会变成原来的2的5次方分之一，需要输入图像是2的5次方的倍数
            w = math.ceil(w / factor) * factor // 2
            #print(h, w)
            mask = np.zeros(shape=(h, w))

            image = cv2.resize(image, (w, h))
            image_ori = image
            # image=illum(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            aug = Compose([
                # PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),  # padding到2的5次方的倍数
                Normalize(p=1)  # 归一化
            ])
            augmented = aug(image=image, mask=mask)
            image = augmented['image']
            image = img_to_tensor(image).unsqueeze(0).to(DEVICE)  # torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # 图像转为tensor格式
            output = model(image)  # 预测
            seg_mask = (output[0].data.cpu().numpy().argmax(axis=0)).astype(np.uint8)
            t2 = time.time()
            print("time:", (t2 - t1))

            full_mask = np.zeros((h, w, 3))
            for mask_label, sub_color in enumerate(class_color):
                full_mask[seg_mask == mask_label, 0] = sub_color[2]
                full_mask[seg_mask == mask_label, 1] = sub_color[1]
                full_mask[seg_mask == mask_label, 2] = sub_color[0]
            # print(full_mask.max())
            # import matplotlib.pyplot as plt
            # plt.imshow(full_mask)
            # plt.show()

            seg = mask_overlay(image_ori, (full_mask > 0)).astype(np.uint8)

            # cv2.imshow("seg",seg)
            # cv2.waitKey(0)
            cv2.imwrite(result_path + "/" + str(i).zfill(4) + ".png", seg)

        save_video(data_path, result_path, h, w, result_name)  # 可以播放保存的视频展示分割结果

    elif type == "video":
        # 对视频进行分割
        splitFrames(data_path, result_path)
        #  调用图像分割的方法分割从视频中保存的图像



    elif type == "nii":
        # 对nii3维医学图像进行分割
        import nibabel as nib

        img = np.array(nib.load(data_path).get_data())
        pass

if __name__ == "__main__":
    # seg = "binary"
    # type = "image"
    # data_path = r"..\images1"
    # result_path = r"..\result"
    main(seg, type, data_path, result_path, result_name)