# fc层用模型前向的梯度和CAM回归的梯度加权更新
import torch.nn.functional
import os
import numpy as np
import cv2
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from Dataset_CIFAR10 import *
from utils.util import *
from VGG16addGAP import *
import user_define_lay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = '../dataset/cifar-10-batches-py'

data_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

test_dataset = CIFAR10Dataset('test', PATH)

test_loader1 = DataLoader(test_dataset, batch_size=1, shuffle=False)
test_loader16 = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 加载总模型
model = totalNet()
model.to(device)
model_path = "saved_model/state_dict_model.pth"
model_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(model_dict)

# 固定BN和dropout层，使得偏置参数不随着发生变化
model.eval()

# 获取label的name
classes_name = return_classname(PATH)

# fc层的权重参数
fc_param = list(model.fcblock.fc.parameters())[0].data.cpu().numpy()

# hookGAP之前的特征图
feature_map = []
def hook_forward_feature(module, input, output):
    feature_map.append(output.data.cpu().numpy())
model.fextractblock.register_forward_hook(hook_forward_feature)

test(model, test_loader16, device)

for input, label in test_loader1:
    while(1):
        input, label = input.to(device), label.to(device)
        output = model(input)

        # 输出softmax
        output_softmax = torch.nn.functional.softmax(output, dim=1).data.squeeze()

        # softmax排序
        probs_sort, idx_sort = output_softmax.sort(0, True)

        # 类型转换
        probs_sort, idx_sort = probs_sort.cpu().numpy(), idx_sort.cpu().numpy()

        # 计算CAM
        CAM = returnCAM(feature_map[-1], fc_param, len(list(idx_sort)))

        # label的CAM
        CAM_gray_label = cv2.resize(CAM[label[0]], (img_w, img_h))

        # predict的CAM
        CAM_gray_predict = cv2.resize(CAM[idx_sort[0]], (img_w, img_h))

        x = input[0].cpu().numpy().transpose(1, 2, 0)*255
        x = np.uint8(x)

        # 将输入图片转化为可以显示的图片
        origin_img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

        # 打印预测和label
        for i in range(0, 10):
            print('{:.3f} -> {}({})'.format(probs_sort[i], idx_sort[i], classes_name[idx_sort[i]]))
        print('predict: {}({}) {:.3f}'.format(classes_name[idx_sort[0]], idx_sort[0], probs_sort[0]))
        print('label:   {}({}) {:.3f}\n'.format(classes_name[label[0]], label[0], output_softmax[label[0]]))

        # 将灰度的CAM转化为热力CAM
        colorCAM_predict = cv2.applyColorMap(CAM_gray_predict, cv2.COLORMAP_JET)
        colorCAM_label = cv2.applyColorMap(CAM_gray_label, cv2.COLORMAP_JET)

        # 热力CAM和原图融合
        # result_predict = cv2.addWeighted(colorCAM_predict, 0.5, origin_img, 0.5, 0)
        result_predict = np.uint8(1 * (1 - (CAM_gray_predict/255).reshape(img_w,img_h,1) ** 0.8) * origin_img + ((CAM_gray_predict/255).reshape(img_w,img_h,1) ** 0.8) * np.float32(colorCAM_predict))

        # result_label = cv2.addWeighted(colorCAM_label, 0.5, origin_img, 0.5, 0)
        result_label = np.uint8(1 * (1 - (CAM_gray_label/255).reshape(img_w,img_h,1) ** 0.8) * origin_img + ((CAM_gray_label/255).reshape(img_w,img_h,1) ** 0.8) * np.float32(colorCAM_label))


        # 图片加入文字信息
        cv2.putText(result_predict, 'predict:' + classes_name[idx_sort[0]] + '  {:.3f}'.format(probs_sort[0]),
                    (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(result_label, 'label:' + classes_name[label[0]] + '  {:.3f}'.format(
            probs_sort[list(idx_sort).index(label.cpu().numpy()[0])]), (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        three = np.hstack((origin_img, result_predict, result_label))

        cv2.namedWindow('|origin image|     |predict CAM|     |label CAM|')
        cv2.imshow('|origin image|     |predict CAM|     |label CAM|', three)

        bool, draw_CAM_output = draw_CAM(origin_img, img_w, img_h)

        if bool:  # 如果回车，即使用绘制CAM更新参数
            draw_CAM_32 = cv2.resize(draw_CAM_output,(img_w,img_h)).reshape(img_w, img_h, 1)  # 修改绘CAM的维度为32，32，1
            draw_CAM_32_norm = (draw_CAM_32 - draw_CAM_32.min()) / (draw_CAM_32.max() - draw_CAM_32.min())  # norm 0~1
            mask_input = origin_img * draw_CAM_32_norm  # 绘的CAM和原图像乘 生成 mask的图像
            mask_input = np.uint8(origin_img * draw_CAM_32_norm)  # 转化为可显示的 mask图像

            # cv2.namedWindow('mask input')
            # cv2.imshow('mask input', mask_input)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 加载特征提取block
            fextractblock_dict = model.fextractblock.state_dict()
            model_fextractblock = fextractBlock()
            model_fextractblock.to(device)
            model_fextractblock.load_state_dict(fextractblock_dict)

            # 创建新的模型，用来更新
            model_updata_dict = model.fextractblock.state_dict()
            model_updata = fextractBlock()
            model_updata.to(device)
            model_updata.load_state_dict(model_updata_dict)

            criterion_feature = torch.nn.CrossEntropyLoss()
            optimizer_feature = torch.optim.SGD(model_updata.parameters(), lr=0.00000001, momentum=0.5)

            feature_output = model_updata(input)  # 原图的特征图

            mask_input_tensor = data_transform(mask_input).view(1, 3, img_w,img_h)  # mask图像进行transforms
            mask_input_tensor = mask_input_tensor.to(device)
            mask_output = model_fextractblock(mask_input_tensor)  # mask图像的输出特征

            loss_feature = criterion_feature(feature_output, mask_output)
            optimizer_feature.zero_grad()
            loss_feature.backward()
            # print(mask_input_tensor)
            optimizer_feature.step()

            model.fextractblock.load_state_dict(model_updata.state_dict())  # 把特征提取网络的参数复制到总网络

            # 冻结卷积层，然后前向反向传播得到fc层的梯度
            for para in model.fextractblock.parameters():
                para.requires_grad = False
            criterion_fc = torch.nn.CrossEntropyLoss()
            optimizer_fc = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.5)
            # optimizer_fc = torch.optim.SGD(model.parameters(), lr=0.00003, momentum=0.5)

            output_fc = model(input)
            loss_fc = criterion_fc(output_fc, label)
            optimizer_fc.zero_grad()
            loss_fc.backward()
            # optimizer_fc.step()
            weight_grad_1 = model.fcblock.fc.weight.grad  # 得到梯度

            # 绘制CAM和原生CAM做反向传播得到fc层的梯度
            output2 = model(input)

            weight_GAPtoFC = list(model.parameters())[-1].data.cpu().numpy()
            user_define = user_define_lay.CAM_backprop(torch.from_numpy(weight_GAPtoFC), label.cpu().numpy()[0])
            user_define = user_define.to(device)
            criterion_user_define = torch.nn.MSELoss()
            optimizer_user_define = torch.optim.SGD(user_define.parameters(), lr=0.00003, momentum=0.5)
            input_user_define = torch.from_numpy(feature_map[-1])
            input_user_define = input_user_define.to(device)
            output_user_define = user_define(input_user_define)
            label_user_define = draw_CAM_output.astype(np.float32) / 255
            label_user_define = cv2.resize(label_user_define, (7, 7))
            label_user_define = torch.from_numpy(label_user_define)
            label_user_define = label_user_define.to(device)
            loss_user_define = criterion_user_define(output_user_define, label_user_define)
            optimizer_user_define.zero_grad()
            loss_user_define.backward()
            # optimizer_user_define.step()

            weight_grad_2 = user_define.weight_param.grad  # 得到梯度

            alph = 0.1
            weight_grad = weight_grad_1*alph + weight_grad_2*(1-alph)  # 两个梯度加权
            model.fcblock.fc.weight.grad = weight_grad

            optimizer_fc.step()

            # for para in model.fextractblock.parameters():
            #     para.requires_grad = True

            test(model, test_loader16, device)

        if bool == False:  # 如果ESC，即跳过本张图片
            break
