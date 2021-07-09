from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageTk, Image
import cv2


class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return None, None


class Paint(object):
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'green'

    def __init__(self):
        self.root = Tk()
        self.root.title('Paint')
        self.vid = MyVideoCapture(0)
        self.width = int(self.vid.width) + 100
        self.height = int(self.vid.height)
        self.root.geometry('{}x{}'.format(self.width, self.height))
        self.root.maxsize(self.width, self.height)
        self.root.minsize(self.width, self.height)

        self.paint_tools = Frame(self.root, width=self.width, height=self.height, relief=RIDGE, borderwidth=2)
        self.paint_tools.place(x=0, y=0)

        self.b = Label(self.paint_tools, borderwidth=0, text='brush', font=('verdana', 10, 'bold'))
        self.b.place(x=5, y=40)
        self.brush_button = Button(self.paint_tools, borderwidth=2, command=self.use_brush)
        self.brush_button.place(x=60, y=40)

        self.cl = Label(self.paint_tools, text='color', font=('verdana', 10, 'bold'))
        self.cl.place(x=5, y=70)
        self.color_button = Button(self.paint_tools, borderwidth=2, command=self.choose_color)
        self.color_button.place(x=60, y=70)

        self.e = Label(self.paint_tools, text='eraser', font=('verdana', 10, 'bold'))
        self.e.place(x=5, y=100)
        self.eraser_button = Button(self.paint_tools, borderwidth=2, command=self.use_eraser)
        self.eraser_button.place(x=60, y=100)

        self.pen_size = Label(self.paint_tools, text="Pen Size", font=('verdana', 10, 'bold'))
        self.pen_size.place(x=15, y=250)
        self.choose_size_button = Scale(self.paint_tools, from_=1, to=10, orient=VERTICAL)
        self.choose_size_button.place(x=20, y=150)
        self.choose_size_button.set(5)

        self.c = Canvas(self.root, bg='white', width=1920, height=1080, relief=RIDGE, borderwidth=0)
        self.c.place(x=100, y=0)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.brush_button

        self.delay = 15
        self.update()
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            bg = self.c.create_image(0, 0, image=self.photo, anchor=NW)
            self.c.tag_lower(bg)
        self.root.after(self.delay, self.update)

    def paint(self, event):

        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()
