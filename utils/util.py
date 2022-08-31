import os
import numpy as np
import torch
import cv2

# CIFAR10解包
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def return_classname(PATH):
    # 获取label的name
    classes_name = unpickle(os.path.join(PATH,'batches.meta'))[b'label_names']
    # 类型转换
    for i in range(0, len(classes_name)):
        classes_name[i] = str(classes_name[i], encoding='utf-8')
    return classes_name

# 计算CAM
def returnCAM(feature_conv, weight_GAPtoFC, len):
    bz, nc, h, w =feature_conv.shape
    output_cam = []
    for i in range(0, len):
        cam =weight_GAPtoFC[i].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h,w)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam_img = np.uint8(cam * 255)
        output_cam.append(cam_img)
    return output_cam

# 绘画CAM
def draw_CAM(source_img, img_w, img_h):
    # 鼠标按下为真
    drawing = False
    ix, iy = -1, -1

    brush_size = 10
    feather_size = 65

    def draw_circle(event, x, y, flags, param):
        global ix, iy
        nonlocal drawing

        if event == cv2.EVENT_LBUTTONDOWN: # 鼠标按下
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE: # 鼠标移动
            if drawing:
                cv2.circle(img_draw_cam, (x, y), brush_size, (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP: # 鼠标抬起
            drawing = False

    img_draw_cam = np.zeros((img_w, img_h, 1), np.uint8)  # 生成全黑画布
    cv2.namedWindow('draw CAM')
    cv2.setMouseCallback('draw CAM', draw_circle)

    while (1):
        feather = cv2.blur(img_draw_cam, (feather_size, feather_size))   # 模糊
        heatmap_drawcam = cv2.applyColorMap(feather, cv2.COLORMAP_JET)

        # result_drawcam = cv2.addWeighted(heatmap_drawcam, 0.5, source_img, 0.5, 0)
        result_drawcam = np.uint8(1 * (1 - (feather/255).reshape(img_w, img_h, 1) ** 0.8) * source_img + ((feather/255).reshape(img_w,img_h,1) ** 0.8) * np.float32(heatmap_drawcam))

        text = cv2.putText(result_drawcam,
                           "Brush size: " + str(brush_size) + "  feather size: " + str(feather_size), (5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('draw CAM', text)
        k = cv2.waitKey(1)
        if k == 27:  # ESC的ascii值
            print('skip')
            cv2.destroyAllWindows()
            return False, 0
        elif k == 13:  # 回车键的ascii值
            print('modified')
            cv2.destroyAllWindows()
            return True, feather
        elif k == ord('-'):
            brush_size = brush_size - 1
            if brush_size < 1:
                brush_size = 1
        elif k == ord('='):
            brush_size = brush_size + 1
            if brush_size >= 100:
                brush_size = 100
        elif k == ord('r'):
            img_draw_cam[:, :, :] = 0
            print('restart draw')
        elif k == ord('['):
            feather_size = feather_size - 1
            if feather_size < 1:
                feather_size = 1
        elif k == ord(']'):
            feather_size = feather_size + 1
            if feather_size > 300:
                feather_size = 300
        elif k == ord('0'):
            brush_size = 10
            feather_size = 65

def test(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data, label = data
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, dim = 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        print('accuracy on test set: %d %% \n' % (100 * correct / total))