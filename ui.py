from tkinter import *
import tkinter as tk  # 使用Tkinter前需要先导入
from tkinter import filedialog, ttk
import tkinter.messagebox  # 这个是消息框，对话框的关键
from PIL import ImageTk
from PIL import Image
import math
import cv2
import threading
import random


def change_type():
    global type
    type = 1 - type


def mouse_left_button_down_event(event):
    global is_mouse_left_button_down
    print("math.pow(abs(event.x - current_oval_x),2) + math.pow(abs(event.y - current_oval_y),2)",math.sqrt( math.pow(abs(event.x - current_oval_x),2) + math.pow(abs(event.y - current_oval_y),2) ))
    if math.sqrt( math.pow(abs(event.x - current_oval_x),2) + math.pow(abs(event.y - current_oval_y),2) ) < 6:
        is_mouse_left_button_down = True
    print("mouse down", event.x, event.y)



def mouse_left_button_release_event(event):
    global is_mouse_left_button_down
    is_mouse_left_button_down = False
    print("mouse realease", event.x, event.y)


def mouse_motion(event):
    global current_oval_x,current_oval_y,scale
    text_var = "鼠标当前位置：x:{}, y:{}".format(event.x, event.y)
    var.set(text_var)
    light_value, time_value, view_value = transform(event.x, event.y, scale)
    if light_value>=0 and light_value<=scale and time_value>=0 and time_value<=scale and view_value>=0 and view_value<=scale:
        if is_mouse_left_button_down:
            current_oval_x = event.x
            current_oval_y = event.y
            canvas.delete("oval")
            canvas.delete("light_text")
            canvas.delete("time_text")
            canvas.delete("view_text")
            canvas.create_oval(current_oval_x - 4, current_oval_y - 4, current_oval_x + 4, current_oval_y + 4, fill="black",tag="oval")
            canvas.create_text(510, 270, text='Light:'+str(light_value), tag="light_text")
            canvas.create_text(510, 290, text='Time:'+str(time_value), tag="time_text")
            canvas.create_text(510, 310, text='View:'+str(view_value), tag="view_text")

def transform(x,y,scale):
    light_value,time_value,view_value = 0,0,0
    length = triangle_y1 - triangle_y0
    light_value = round(1 - (y - triangle_y0) / length * scale,2)
    time_temp1 = (x-440)/math.sqrt(3)*2 - ((x-440)/math.sqrt(3)  - (triangle_y1 - y))/2
    time_value = round(1 - time_temp1 / length * scale ,2 )
    view_temp1 = (580-x) / math.sqrt(3) * 2 - ((580-x) / math.sqrt(3) - (triangle_y1 - y)) / 2
    view_value = round(1 - view_temp1 / length * scale ,2 )
    return light_value,time_value,view_value

# 鼠标左键是否处于按下框选状态
is_mouse_left_button_down = False
mouse_left_button_start_pos_x, mouse_left_button_start_pos_y = 0, 0
current_oval_x = 510
current_oval_y = 200
scale = 1
# UI生成———————————————————————————————————————————————————————————————————————————————————————————————————————————
# 生成窗口
root = Tk()
root.title("XField")
root.minsize(800, 500)	   # 最小尺寸
root.maxsize(800, 500)    # 最大尺寸

# 生成introduction文字提示信息
introduction = Label(root, text="使用说明：XFiled-Pytorch复现", font=('Arial', 14))
introduction.pack()

# 生成canvas画布
canvas = Canvas(root, width=600, height=400, background='white')
canvas.pack(anchor=CENTER, side=TOP, padx=0, pady=0)

canvas.create_line(20, 60, 400, 60)
canvas.create_line(20, 60,20,340)
canvas.create_line(20,340,400, 340)
canvas.create_line(400, 340, 400, 60)

trianle_center_y = 200
triangle_y0 = 200-140/math.sqrt(3)
triangle_y1 = 200+70/math.sqrt(3)
canvas.create_line(440, triangle_y1, 580, triangle_y1)
canvas.create_line(580, triangle_y1, 510, triangle_y0)
canvas.create_line(510, triangle_y0 , 440, triangle_y1)

canvas.create_line(510, trianle_center_y, 440, triangle_y1,fill="red",dash=(3, 5))
canvas.create_line(510, trianle_center_y, 580, triangle_y1,fill="red",dash=(3, 5))
canvas.create_line(510, trianle_center_y , 510, triangle_y0,fill="red",dash=(3, 5))

x0 = 510
y0 = 200
canvas.create_oval(x0 - 4, y0 - 4, x0 + 4, y0 + 4,fill="black",tag="oval")

canvas.create_text(510, 100, text='Light')
canvas.create_text(430, 250, text='Time')
canvas.create_text(585, 250, text='View')
canvas.create_text(510, 270, text='Light:0.3',tag="light_text")
canvas.create_text(510, 290, text='Time:0.3',tag="time_text")
canvas.create_text(510, 310, text='View:0.3',tag="view_text")

# 鼠标位置实时显示模块生成
x, y = 0, 0
var = StringVar()
text = "鼠标当前位置：x:{}, y:{}".format(x, y)
var.set(text)
lab = Label(root, textvariable=var)
lab.pack(anchor=S, side=RIGHT, padx=10, pady=10)


canvas.bind("<Motion>", mouse_motion)
canvas.bind("<Button-1>", mouse_left_button_down_event)
canvas.bind("<ButtonRelease-1>", mouse_left_button_release_event)


mainloop()
