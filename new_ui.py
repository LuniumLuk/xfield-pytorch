from tkinter import *
import tkinter as tk  # 使用Tkinter前需要先导入
from tkinter import filedialog, ttk
import tkinter.messagebox  # 这个是消息框，对话框的关键
from PIL import ImageTk
from PIL import Image
import numpy as np
import math
import cv2
from easydict import EasyDict
import threading
import random
from xfield_test import XFieldTest


def mouse_left_button_down_event(event):
    global is_mouse_left_button_down, mouse_moving_oval_index
    if math.sqrt(math.pow(abs(event.x - oval_x1), 2) + math.pow(abs(event.y - oval_y1), 2)) < 5:
        is_mouse_left_button_down = True
        mouse_moving_oval_index = 1

    elif math.sqrt(math.pow(abs(event.x - oval_x2), 2) + math.pow(abs(event.y - oval_y2), 2)) < 5:
        is_mouse_left_button_down = True
        mouse_moving_oval_index = 2

    print("mouse down", event.x, event.y)


def mouse_left_button_release_event(event):
    global is_mouse_left_button_down
    is_mouse_left_button_down = False
    print("mouse realease", event.x, event.y)
    canvas.create_image(210, 200, image=result_photo, tag="image")
    canvas.update()


def mouse_motion(event):
    global oval_x1, oval_y1, oval_x2, oval_y2, light_value, view_value, time_value, result_photo
    text_var = "鼠标当前位置：x:{}, y:{}".format(str(event.x).zfill(3), str(event.y).zfill(3))
    var.set(text_var)
    if is_mouse_left_button_down:
        if mouse_moving_oval_index == 1:
            canvas.delete("oval1")
            canvas.delete("light_text")
            canvas.delete("view_text")
            oval_x1 = event.x
            oval_y1 = event.y
            if oval_x1 < 440:
                oval_x1 = 440
            elif oval_x1 > 580:
                oval_x1 = 580
            if oval_y1 < 100:
                oval_y1 = 100
            elif oval_y1 > 240:
                oval_y1 = 240
            canvas.create_oval(oval_x1 - 4, oval_y1 - 4, oval_x1 + 4, oval_y1 + 4, fill="black", tag="oval1")
            light_value, view_value = transformOval1()
            canvas.create_text(510, 340, text='Light:' + str(light_value), tag="light_text")
            canvas.create_text(510, 360, text='View:' + str(view_value), tag="view_text")
        elif mouse_moving_oval_index == 2:
            # 仅允许在横轴上运动
            canvas.delete("oval2")
            canvas.delete("time_text")
            oval_x2 = event.x
            if oval_x2 < 440:
                oval_x2 = 440
            elif oval_x2 > 580:
                oval_x2 = 580
            canvas.create_oval(oval_x2 - 4, oval_y2 - 4, oval_x2 + 4, oval_y2 + 4, fill="black", tag="oval2")
            time_value = transformOval2()
            canvas.create_text(510, 380, text='Time:' + str(time_value), tag="time_text")
        img = x.generateResult(light_value, view_value, time_value)
        img = cv2.resize(img, (380, 280))
        result_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result_photo = ImageTk.PhotoImage(result_image)
        canvas.create_image(210, 200, image=result_photo, tag="image")
        canvas.update()


def transformOval1():
    light_value = round(max_coordinate - (oval_y1 - 100) / 140 * max_coordinate, 2)
    view_value = round((oval_x1 - 440) / 140 * max_coordinate, 2)
    return light_value, view_value


def transformOval2():
    time_value = round((oval_x2 - 440) * max_coordinate / 140, 2)
    return time_value


def resize(event):
    global oval_x1, oval_y1, oval_x2, oval_y2, time_value, light_value, view_value
    global max_coordinate  # 最大坐标
    old_max_coordinate = max_coordinate
    max_coordinate = scaleLine.get()
    oval_x1 = (oval_x1 - 440) * old_max_coordinate / max_coordinate + 440
    oval_y1 = 240 - (240 - oval_y1) * old_max_coordinate / max_coordinate
    oval_x2 = (oval_x2 - 440) * old_max_coordinate / max_coordinate + 440
    if oval_x1 > 580:
        oval_x1 = 580
        view_value = max_coordinate
    if oval_y1 < 100:
        oval_y1 = 100
        light_value = max_coordinate
    if oval_x2 > 580:
        oval_x2 = 580
        time_value = max_coordinate
    canvas.delete("light_text")
    canvas.delete("time_text")
    canvas.delete("view_text")
    canvas.delete("scale_text_x")
    canvas.delete("scale_text_y")
    canvas.delete("scale_text")
    canvas.delete("oval1")
    canvas.delete("oval2")
    canvas.create_oval(oval_x1 - 4, oval_y1 - 4, oval_x1 + 4, oval_y1 + 4, fill="black", tag="oval1")
    canvas.create_oval(oval_x2 - 4, oval_y2 - 4, oval_x2 + 4, oval_y2 + 4, fill="black", tag="oval2")
    canvas.create_text(510, 340, text='Light:' + str(light_value), tag="light_text")
    canvas.create_text(510, 360, text='View:' + str(view_value), tag="view_text")
    canvas.create_text(510, 380, text='Time:' + str(time_value), tag="time_text")
    canvas.create_text(585, 315, text=str(max_coordinate), tag="scale_text_x")
    canvas.create_text(430, 95, text=str(max_coordinate), tag="scale_text_y")
    canvas.create_text(585, 250, text=str(max_coordinate), tag="scale_text")


def cmb_select(event):
    pb.start()
    th = threading.Thread(target=cmb_select_main, args=(cmb.get(),))
    th.start()
    tk.messagebox.showinfo('进行', '开始加载数据请耐心等待')

def delete_all():
    canvas.delete("time_text")
    canvas.delete("light_text")
    canvas.delete("view_text")
    canvas.delete("oval1")
    canvas.delete("oval2")
    canvas.delete("2-dimension-box")
    canvas.delete("1-dimension-line")
    canvas.delete("scale_text_x")
    canvas.delete("scale_text_y")
    canvas.delete("scale_text")
    canvas.delete("x_text")
    canvas.delete("y_text")
    canvas.delete("text")
    canvas.delete("zero1")
    canvas.delete("zero2")
def draw_all():
    # 绘制两个0
    canvas.create_text(430, 250, text='0', tag="zero1")
    canvas.create_text(430, 315, text='0', tag="zero2")
    # 绘制放缩栏
    canvas.create_text(585, 315, text=str(max_coordinate), tag="scale_text")
    canvas.create_text(430, 95, text=str(max_coordinate), tag="scale_text_y")
    canvas.create_text(585, 250, text=str(max_coordinate), tag="scale_text_x")
    # 绘制文字
    canvas.create_text(440, 80, text='Light', tag="y_text")
    canvas.create_text(585, 265, text='View', tag="x_text")
    canvas.create_text(585, 330, text='Time', tag="text")
    # 绘制指示值
    canvas.create_text(510, 340, text='Light:0', tag="light_text")
    canvas.create_text(510, 360, text='View:0', tag="view_text")
    canvas.create_text(510, 380, text='Time:0', tag="time_text")
    # 二维空间
    canvas.create_line(440, 100, 580, 100, tag="2-dimension-box")
    canvas.create_line(580, 100, 580, 240, tag="2-dimension-box")
    canvas.create_line(580, 240, 440, 240, tag="2-dimension-box")
    canvas.create_line(440, 240, 440, 100, tag="2-dimension-box")

    # 单维轴
    canvas.create_line(440, 300, 580, 300, tag="1-dimension-line")
    # 第一个滑块
    canvas.create_oval(oval_x1 - 4, oval_y1 - 4, oval_x1 + 4, oval_y1 + 4, fill="black", tag="oval1")
    # 第二个滑块
    canvas.create_oval(oval_x2 - 4, oval_y2 - 4, oval_x2 + 4, oval_y2 + 4, fill="black", tag="oval2")


def cmb_select_main(dataset):
    global x, canvas, result_photo
    delete_all()
    if dataset == "apple-3-dimension":
        draw_all()  #绘制所有canvas内容
        args = EasyDict({
            'dataset': './data/3x3x3/apple',
            'savedir': './results/3x3x3/apple',
            'type': ['light', 'time', 'view'],
            'dims': [3, 3, 3],
            'DSfactor': 8,
            'neighbor_num': 4,
            'lr': 0.0001,
            'sigma': 0.1,
            'stop_l1_thr': 0.01
        })
        num_n = 8  # 生成用的邻居数量
        scale = 90
        x = XFieldTest(args, num_n, scale)
        img = x.generateResult(light_value, view_value, time_value)
        img = cv2.resize(img, (380, 280))
        result_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result_photo = ImageTk.PhotoImage(result_image)
        canvas.create_image(210, 200, image=result_photo, tag="image")
        canvas.update()
    elif dataset == "apple-1-dimension-time":
        # 第二个滑块
        canvas.create_oval(oval_x2 - 4, oval_y2 - 4, oval_x2 + 4, oval_y2 + 4, fill="black", tag="oval2")
        # 单维轴
        canvas.create_line(440, 300, 580, 300, tag="1-dimension-line")
        canvas.create_text(510, 380, text='Time:0', tag="time_text")
        canvas.create_text(585, 330, text='Time', tag="text")
        canvas.create_text(585, 315, text=str(max_coordinate), tag="scale_text")
        canvas.create_text(430, 315, text='0', tag="zero2")

    pb.stop()
    tk.messagebox.showinfo('成功', '数据加载完成')


# 鼠标左键是否处于按下框选状态
result_photo = None
is_mouse_left_button_down = False
# 模式维度
mode = ["light","view","time"]
# 当前移动的是哪个oval
mouse_moving_oval_index = 1
max_coordinate = 1
light_value, view_value, time_value = 0, 0, 0
# UI生成———————————————————————————————————————————————————————————————————————————————————————————————————————————
# 生成窗口
root = Tk()
root.title("XField")
root.minsize(800, 600)	   # 最小尺寸
root.maxsize(800, 600)    # 最大尺寸

# 生成introduction文字提示信息
introduction = Label(root, text="XField-Pytorch,请在下方选取数据集", font=('Arial', 14))
introduction.pack()

# 设置下拉菜单
cmb = ttk.Combobox(root)
cmb.pack()
# 设置下拉菜单中的值
cmb['value'] = ('mydata','apple-3-dimension', 'apple-1-dimension-time')
# 设置默认值，即默认下拉框中的内容
cmb.current(0)

# 生成canvas画布
canvas = Canvas(root, width=600, height=400, background='white')
canvas.pack(anchor=CENTER, side=TOP, padx=0, pady=0)

oval_x1 = 440
oval_y1 = 240
oval_x2 = 440
oval_y2 = 300

# Scale
frame = LabelFrame(root, height=60, width=150, text='改变最大坐标Scale')
frame.pack(side='left', fill='none', expand=True)

scaleLine = tk.Scale(frame, from_=1, to=10, resolution=1, orient=tk.HORIZONTAL, command=resize)
scaleLine.set(1)  # 设置初始值
scaleLine.grid(row=0, column=0)

draw_all()

# 鼠标位置实时显示模块生成
x, y = 0, 0
var = StringVar()
text = "鼠标当前位置：x:{}, y:{}".format(str(x).zfill(3), str(y).zfill(3))
var.set(text)


lab = Label(root, textvariable=var)
lab.pack(anchor=S, side=RIGHT, padx=10, pady=10)
pb = ttk.Progressbar(root, length=400, value=0, mode="indeterminate")
pb.pack(pady=10)

# 初始参数选择与加载绘制
args = EasyDict({
    'dataset': './data/3x3x3/mydata',
    'savedir': './results/3x3x3/mydata',
    'type': ['light', 'time', 'view'],
    'dims': [3, 3, 3],
    'DSfactor': 16,
    'neighbor_num': 2,
    'lr': 0.0001,
    'sigma': 0.1,
    'stop_l1_thr': 0.01
    })
num_n = 8  # 生成用的邻居数量
scale = 90

x = XFieldTest(args, num_n, scale)
img = x.generateResult(light_value, view_value, time_value)
img = cv2.resize(img, (380, 280))
result_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
result_photo = ImageTk.PhotoImage(result_image)
canvas.create_image(210, 200, image=result_photo, tag="image")
canvas.update()


canvas.bind("<Motion>", mouse_motion)
canvas.bind("<Button-1>", mouse_left_button_down_event)
canvas.bind("<ButtonRelease-1>", mouse_left_button_release_event)

cmb.bind("<<ComboboxSelected>>", cmb_select)

mainloop()
