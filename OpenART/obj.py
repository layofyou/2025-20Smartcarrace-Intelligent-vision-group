import gc
import sensor, image, time, os, tf
import pyb, machine
import math
from machine import UART
from machine import Pin # 从pyb导入Pin

# 单独调试专用 1调试 0作为文件调用
debug = 1



uart = machine.UART(1,115200)   # 定义串口1，波特率115200

#*********************神经网络********************#
net_path = "*************.tflite"
labels = [line.rstrip() for line in open("/sd/labels_number.txt")]   # 加载标签 "labels_number.txt"
net = tf.load(net_path, load_to_fb=True)
#************************************************#

#*********************参数定义********************#
value=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
mask = 0
x_1=x_2=x_3=0
Cmark = 0
Dmark = 0



def RGNN_IMG():

    sensor.reset() #初始化感光元件
    sensor.set_pixformat(sensor.RGB565) #设置为RGB565色彩空间
    sensor.set_framesize(sensor.QVGA) #设置图像的大小
    sensor.set_auto_whitebal(False) #自动白平衡关闭
    sensor.set_auto_exposure(False, exposure_us = 1100) #自动曝光关闭
    sensor.set_brightness(700) # 设置图像亮度 越大越亮
    sensor.set_auto_gain(True)  # must turn this off to prevent image washout...
    #sensor.set_auto_whitebal(True,(0,0x80,0))  # must turn this off to prevent image washout...
    sensor.skip_frames(time = 1000) #等待1000ms使感光元件稳定




# 滤波器
def lvboqi():
    print("滤波")


expand_xy = 1.2 # 膨胀系数
expand_w = 1.8 # 膨胀系数
expand_h = 1.3 # 膨胀系数
def de_num(t):

    center_x  = t[0] + t[2]/2
    center_y  = t[1] + t[3]/2
    print("中心")
    print(center_x)
    print(center_y)
    # 前两个数除以1.2并转化为整数
    first_two_scaled = tuple(int(x / expand_xy) for x in t[:2])

    # 后两个数乘以1.2并转化为整数（使用round函数可以更精确地四舍五入到最近的整数）
    # 如果你想要向下取整，可以继续使用int()
    last_w_scaled = tuple(int(round(x * expand_w)) for x in t[2:3])

    last_h_scaled = tuple(int(round(x * expand_h)) for x in t[3:])
    # 合并结果
    result = first_two_scaled + last_w_scaled + last_h_scaled
    return result

def expand_roi(t,expand =1.4):

    center_x  = t[0] + t[2]/2
    center_y  = t[1] + t[3]/2
    # 计算中心坐标


    # 计算新尺寸
    new_w = expand * t[2] *1.15
    new_h = expand * t[3]

    # 计算新边界
    new_x = int(center_x - new_w / 2)
    new_y = int(center_y - new_h / 2)
    new_w = int(new_w)
    new_h = int(new_h)

    ## 裁剪到图像范围内
    #new_left = max(0.0, new_left)
    #new_right = min(float(320, new_right)
    #new_top = max(0.0, new_top)
    #new_bottom = min(float(240), new_bottom)

    ## 防止负尺寸
    #new_w = max(0.0, new_right - new_left)
    #new_h = max(0.0, new_bottom - new_top)

    return (new_x, new_y, new_w, new_h)

roi = [30,30,260,180]
thresholds = (0, 177) # 白色是追踪值
#thresholds = (64, 85, -4, 55, -6, 23)
def find_digits_roi(img):
    img_roi = img.copy()
    #img_roi.dilate(2)
    num_roi = []
    for blob in img_roi.find_blobs([thresholds],roi= roi, pixels_threshold=500, area_threshold=500, merge=True):
        rect = blob.rect()
        #img.draw_rectangle(rect, color=(255, 0, 0))
        rect =  expand_roi(rect)
        num_roi.append(rect)
        #img.draw_rectangle(rect, color=(255, 0, 255))

    return sorted(num_roi, key=lambda x: x[0])


def obj_Find():
    img_gray = sensor.snapshot().to_grayscale().lens_corr(1.8)  # 转换为灰度图
    rect = find_digits_roi(img_gray)  # 查找数字区域
    #img_gray = sensor.snapshot()
    img = sensor.snapshot()
    img.lens_corr(1.8) # 畸变校正
    img = img.to_grayscale() .binary([thresholds]).invert()


    num = []  # 初始化数字列表
    for n, r in enumerate(rect[:5]):  # 只处理前5个矩形,防止内存溢出，最大只能识别5位数
        if n >= 2:
            break  # 这个break在这里不是必需的，但它强调了循环的限制
        img.draw_rectangle(r, color=(255, 0, 0))  # 绘制矩形框
        cropped = img.copy(roi=r) #裁剪出目标区域的图像。

        # 计算缩放比例并缩放为 28x28
        original_width = cropped.width()
        original_height = cropped.height()
        scale_x = 32.0 / original_width
        scale_y = 32.0 / original_height
        cropped_img = cropped.scale(scale_x, scale_y)


        #for obj in tf.classify(net, img, r, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        for obj in tf.classify(net,cropped_img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
            sorted_list = sorted(zip(labels, obj.output()), key=lambda x: x[1], reverse=True)
            recognized_num = sorted_list[0][0]
            num.append(recognized_num)  # 使用append添加元素
            #print(sorted_list)
            #print(f"Recognized number: {recognized_num}")
            img.draw_string(r[0], r[1], f"{recognized_num}", color=(255, 0, 0), scale=2)

    try:
        # 将列表中的每个数字转换成字符串，并连接它们
        num_str = ''.join(map(str, num))
        # 将连接后的字符串转换成整数
        num_int = int(num_str)
        #print(f"Recognized numbers: {num}")
        #print(f"Recognized numbers: {num_int}")
        del img,img_gray,num,num_str
        return num_int
    except :
        return -1

def identify():
    ##*********************主循环********************#
        ##img = sensor.snapshot()
        #img_gray = sensor.snapshot().to_grayscale()  #灰度
        #find_digits_roi(img_gray)
        ##img_gray = sensor.snapshot().to_grayscale()  #灰度
        result = obj_Find()
        print(f"Recognized numbers: {result}")
        #print(clock.fps(), "fps")
        try :
            uart.write(bytes([0x1B, result, 0xFF]))
            print("物品识别：发送完毕！")
        except :
             print("物品识别：发送失败！")

        del result
        gc.collect()

        return 1 # 启用串口



if debug == 1 :
    RGNN_IMG()
    #*************************************************************************
    get_thresholds = 0
    #*************************************************************************
    if get_thresholds :
        print("获取阈值")
        while True:
            #img_gray = sensor.snapshot()
            img_gray = sensor.snapshot().to_grayscale()  # 转换为灰度图
    clock = time.clock()
    try:
        while True:
            clock.tick()
            ##img = sensor.snapshot()
            #img_gray = sensor.snapshot().to_grayscale()  #灰度
            #find_digits_roi(img_gray)
            ##img_gray = sensor.snapshot().to_grayscale()  #灰度
            result = obj_Find()
            print(f"Recognized numbers: {result}")


            try :
                uart.write(bytes([0x1B, result, 0xFF]))
                print("物品识别：发送完毕！")
            except :
                 print("物品识别：发送失败！")
            print(clock.fps(), "fps")
            del result
            gc.collect()
    except KeyboardInterrupt:
        print("User stop")
    except Exception as e:
        print(f"Exception: {e}")
