

import pyb, machine, gc
import sensor, image, time, math
import os, tf
from machine import UART
from machine import Pin




sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((160, 160))
sensor.set_auto_whitebal(False)
sensor.set_auto_exposure(False,exposure_us =1100)
sensor.set_brightness(800)
sensor.set_auto_gain(True)
sensor.set_auto_whitebal(True,(0,0x80))
sensor.skip_frames(time = 7000)

clock =time.clock()


debug = 0




first_calss = 0


uart = machine.UART(2,115200)





net_path_a = "big_V421.tflite" # 定义模型的路
labels = [line.rstrip() for line in open("/sd/big_V421.txt")]   # 加载标签
big_net = tf.load(net_path_a, load_to_fb=True)


current_mode = None
hand_net_path = None
hand_labels = None
obj_net_path = None
obj_labels = None

# 模式控制函数
def mode_1():
    """物品识别模式（基于手写数字识别代码结构改写）"""
    global current_mode
    current_mode = "物品识别"
    print("进入模式 1：物品识别")

    # 物品ID映射表
    object_id_map = {
        'wrench': 102, 'soldering_iron': 104, 'electrodrill': 106, 'tape_measure': 108,
        'screwdriver': 110, 'pliers': 112, 'oscillograph': 114, 'multimeter': 116,
        'printer': 101, 'keyboard': 103, 'mobile_phone': 105, 'mouse': 107,
        'headphones': 109, 'monitor': 111, 'speaker': 113
    }

    # 模型加载（参考手写数字识别的模型加载方式）
    print("物品模型加载中...")
    while True:
        try:
            object_net_path = "mobilenet_laoge.tflite"
            object_labels = [line.rstrip() for line in open("/sd/laoge.txt")]
            object_net = tf.load(object_net_path, load_to_fb=True)
            if object_net is not None:
                print("物品模型加载成功！")
                break
        except Exception as e:
            print(f"物品模型加载失败: {e}")
            time.sleep(1)  # 防止频繁尝试

    # 图像预处理函数（简化版）
    def preprocess_image():
        img = sensor.snapshot().lens_corr(1.8)  # 单行完成拍摄和畸变校正
        return img

    # 物品检测与处理函数（整合了检测、绘制和发送）
    def object_detect_and_process():
        img = preprocess_image()
        objects = []

        # 运行模型推理（参考手写数字的TF分类方式）
        for obj in object_net.classify(img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
            sorted_list = sorted(zip(object_labels, obj.output()), key=lambda x: x[1], reverse=True)
            if sorted_list[0][1] >= 0.6:  # 置信度阈值
                obj_name = sorted_list[0][0]
                objects.append({
                    'name': obj_name,
                    'confidence': sorted_list[0][1],
                    'box': obj.rect()
                })

                # 直接在分类循环中绘制（减少循环次数）
                x, y, w, h = obj.rect()
                img.draw_rectangle(x, y, w, h, color=(255, 0, 0), thickness=2)
                img.draw_string(x, y-10, f"{obj_name}:{sorted_list[0][1]:.2f}", color=(255,0,0), scale=1.5)

        # 串口发送（参考手写数字的串口发送方式）
        if not objects:
            uart.write('@' + str(0xFF) + '\n')
        else:
            for obj in objects:
                if obj['name'] in object_id_map:
                    packet = '@' + object_id_map[obj['name']],int(obj['confidence'] * 100) + '\n'
                    uart.write(packet)
                    time.sleep(0.05)

        # 显示帧率和物体数量（参考手写数字的调试信息）
        img.draw_string(0, 0, f"FPS:{clock.fps():.1f}", color=(255,0,0))
        img.draw_string(0, 20, f"Objs:{len(objects)}", color=(255,0,0))

        if debug and objects:
            print("检测结果:", [(o['name'], o['confidence']) for o in objects])

        gc.collect()
        return 1 if objects else 0




def mode_2():
    """手写数字识别模式"""
    global current_mode
    current_mode = "手写数字识别"
    print("进入模式 2：手写数字识别")






    while (True):

        print("模型加载中")

        try:
            hand_net_path = "write_num_V3.tflite"
            hand_labels = [line.rstrip() for line in open("/sd/labels_number.txt")]  # 加载标签 "labels_number.txt"
            hand_net = tf.load(hand_net_path, load_to_fb=True)
            if hand_net is None:
                continue  # 进入下一轮循环
            print("模型加成功！")
        except:
            print("模型加载失败！")
        if hand_net != None:
            break




        ############################################################################

        # 图片旋转 1 0度 2 90度 3 4

        # 图片旋转 1 0度 2 90度 3 4
        def rotate(num, img):
            if num == 1:
                # 0 degrees: no changes
                return img.replace(vflip=False, hmirror=False, transpose=False)
            elif num == 2:
                # 90° clockwise: transpose + horizontal flip
                return img.replace(vflip=True, hmirror=False, transpose=True)
            elif num == 3:
                # 180°: vertical + horizontal flip
                return img.replace(vflip=True, hmirror=True, transpose=False)
            elif num == 4:
                # 270° clockwise: transpose + vertical flip
                return img.replace(vflip=False, hmirror=True, transpose=True)
            else:
                raise ValueError("Invalid rotation number. Use 1, 2, 3, or 4.")

        def expand_roi(t, expand=1.4):

            center_x = t[0] + t[2] / 2
            center_y = t[1] + t[3] / 2
            # 计算中心坐标

            # 计算新尺寸
            new_w = expand * t[2] * 1.15
            new_h = expand * t[3]

            # 计算新边界
            new_x = int(center_x - new_w / 2)
            new_y = int(center_y - new_h / 2)
            new_w = int(new_w)
            new_h = int(new_h)

            return (new_x, new_y, new_w, new_h)

        hand_roi = [30, 30, 260, 180]
        hand_thresholds = (0, 192)  # 白色是追踪值

        # thresholds = (64, 85, -4, 55, -6, 23)
        def find_digits_roi(img):
            img_roi = img.copy()
            num_roi = []
            for blob in img_roi.find_blobs([hand_thresholds], roi=hand_roi, pixels_threshold=500, area_threshold=500,
                                           merge=True):
                rect = blob.rect()
                # img.draw_rectangle(rect, color=(255, 0, 0))
                rect = expand_roi(rect)
                num_roi.append(rect)
                # img.draw_rectangle(rect, color=(255, 0, 255))

            return sorted(num_roi, key=lambda x: x[0])

        ## 合并区域
        def find_connect_roi(img):
            img_roi = img.copy()
            # 用于保存所有区域的坐标
            regions = []
            for blob in img_roi.find_blobs([hand_thresholds], roi=hand_roi, pixels_threshold=500, area_threshold=500,
                                           merge=True):
                rect = blob.rect()
                rect = expand_roi(rect)  # 假设 expand_roi 返回 (x, y, w, h)
                regions.append(rect)

            # 如果没有检测到区域，直接返回空
            if not regions:
                return None

            # 合并所有区域为一个包围矩形
            min_x = min(r[0] for r in regions)
            min_y = min(r[1] for r in regions)
            max_x = max(r[0] + r[2] for r in regions)
            max_y = max(r[1] + r[3] for r in regions)
            merged_rect = (min_x, min_y, max_x - min_x, max_y - min_y)

            return merged_rect

        def capture_and_preprocess_image(img2, rnum):
            """图像采集与基础预处理"""
            img2 = sensor.snapshot().lens_corr(1.8)  # 畸变校正
            rect2 = find_connect_roi(img2)  ## 查找数字区域
            img1 = img2.copy(roi=rect2)  # 裁剪出目标区域的图像。
            img1 = rotate(rnum, img1)
            img = img1.copy()

            img_gray = img1.to_grayscale()  # .lens_corr(1.8)  # 转换为灰度图
            rect1 = find_digits_roi(img_gray)  # 查找单个数字区域

            img = img.to_grayscale().binary([hand_thresholds]).invert()
            return img, rect1

        #  缩放图片
        def crop_and_scale_region(region, r, target_size):
            """裁剪并缩放区域"""
            cropped = region.copy(roi=r)
            scale_x = target_size[0] / cropped.width()
            scale_y = target_size[1] / cropped.height()
            return cropped.scale(scale_x, scale_y)

        def RGNN_Find(hand_net=None, hand_labels=None, rnum=1):

            img = sensor.snapshot()
            img, rect = capture_and_preprocess_image(img, rnum)

            num = []  # 初始化数字列表
            score = []  # 初始化数字列表
            for n, r in enumerate(rect[:5]):  # 只处理前5个矩形,防止内存溢出，最大只能识别5位数
                if n >= 2:
                    break  # 这个break在这里不是必需的，但它强调了循环的限制
                img.draw_rectangle(r, color=(255, 0, 0))  # 绘制矩形框

                cropped_img = crop_and_scale_region(img, r, [32, 32])

                # for obj in tf.classify(net, img, r, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
                for obj in tf.classify(hand_net, cropped_img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5,
                                       y_overlap=0.5):
                    sorted_list = sorted(zip(hand_labels, obj.output()), key=lambda x: x[1], reverse=True)
                    recognized_num = sorted_list[0][0]
                    num.append(recognized_num)  # 使用append添加元素
                    recognized_score = sorted_list[0][1]
                    score.append(recognized_score)
                    # print(sorted_list)
                    # print(f"Recognized number: {recognized_num}")
                    img.draw_string(r[0], r[1], f"{recognized_num}", color=(255, 0, 0), scale=2)

            try:
                # 将列表中的每个数字转换成字符串，并连接它们
                num_str = ''.join(map(str, num))
                # 将连接后的字符串转换成整数
                num_int = int(num_str)

                del img, num, num_str
                gc.collect()  # 显式触发垃圾收集，尝试回收内存
                return num_int, sum(score)
            except:
                print("数字拼接失败！")

                return -1, -1

        def hand_identify():

            # 主循环函数，完成手写数字识别和结果输出

            # 定义角度与旋转标识的映射关系
            rotations = [
                (4, "270"),  # 旋转4次对应270度
                (3, "180"),
                (2, "90"),
                (1, "0")
            ]

            results = {}  # 存储各角度识别结果

            for rnum, angle_desc in rotations:
                try:
                    num, score = RGNN_Find(hand_net, hand_labels, rnum)
                    if num != -1 and score > 0:
                        results[angle_desc] = (num, score)
                        # print(f"Recognized {angle_desc} numbers: {num}, Score: {score:.2f}")
                    else:
                        print(f"No valid digits recognized at {angle_desc} degrees")
                except Exception as e:
                    print(f"Error processing {angle_desc} degree image: {e}")

            # 选择最优结果
            if not results:
                print("No valid recognition results")
                return -1  # 无有效识别结果

            max_score = max(score for _, score in results.values())
            best_results = [res for res in results.values() if res[1] == max_score]

            # 处理平局情况（取数值最大的数字）
            best_num = max(res[0] for res in best_results)
            best_angle = next(angle for angle, (_, score) in results.items() if score == max_score)

            print(f"\nBest result: {best_num} ({best_angle} degrees), Score: {max_score:.2f}")

            # 串口输出优化
            output_packet = bytearray("@best_num\n")
            try:
                uart.write(output_packet)
                print("手写数字识别：发送完毕！")
            except OSError as e:
                print(f"串口写入失败: {e}")

            # 清理资源
            gc.collect()

            return 1  # 执行成功,回归串口





while True:
    img = sensor.snapshot()
    clock.tick()

    # 如果不在任何模式中，运行模型推理以决定进入哪个模式
    if current_mode is None:
        for obj in big_net.classify(img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
            # 获取预测结果（类别索引和置信度）
            class_id = obj.output().index(max(obj.output()))
            confidence = max(obj.output())

            # 只在置信度足够高时才切换模式
            if confidence > 0.8:
                print("检测到: %s (置信度: %.1f%%)" % (labels[class_id], confidence * 100))

                # 根据类别切换模式
                if class_id == 0:  # 物品
                    mode_1()
                elif class_id == 1:  # 手写数字
                    mode_2()
                else:  # 其他
                    print("其他类别，不切换模式")

    # 显示帧率和当前模式
    status_text = f"FPS: {clock.fps():.1f}"
    if current_mode:
        status_text += f" | Mode: {current_mode}"

    img.draw_string(0, 0, status_text, color=(255, 0, 0))

