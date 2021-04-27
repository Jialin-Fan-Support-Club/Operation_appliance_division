
from PySide2.QtGui import QPixmap,QImage
from PySide2.QtUiTools import QUiLoader
import cv2
from PySide2.QtCore import Signal, QObject,QTimer

class ShowResultWin():  # 展示结果窗口
    def __init__(self,resultpath):
        super().__init__()
        self.ui = QUiLoader().load('ui/show_result_win.ui')
        # 这里需要有代码能够去读主窗口中放置结果的路径，并打开存下的视频
        self.file=resultpath
        if not self.file:
            return
        self.ui.LoadingInfo.setText("正在读取请稍后...")
        # 设置时钟
        self.v_timer = QTimer() #self
        # 读取视频
        self.cap = cv2.VideoCapture(self.file)
        if not self.cap:
            print("打开视频失败")
            return
        # 获取视频FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) # 获得码率
        # 获取视频总帧数
        self.total_f = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 获取视频当前帧所在的帧数
        self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 设置定时器周期，单位毫秒
        self.v_timer.start(int(1000 / self.fps))
        print("FPS:".format(self.fps))
        # 连接定时器周期溢出的槽函数，用于显示一帧视频
        self.v_timer.timeout.connect(self.show_pic)
        # 连接按钮和对应槽函数，lambda表达式用于传参
        self.ui.play.clicked.connect(self.go_pause)
        self.ui.replay.clicked.connect(self.replay)
        self.ui.back.pressed.connect(lambda: self.last_img(True))
        self.ui.back.clicked.connect(lambda: self.last_img(False))
        self.ui.forward.pressed.connect(lambda: self.next_img(True))
        self.ui.forward.clicked.connect(lambda: self.next_img(False))
        print("init OK")

    # 视频播放
    def show_pic(self):
        # 读取一帧
        success, frame = self.cap.read()
        if success:
            # Mat格式图像转Qt中图像的方法
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.ui.Video_Label.setPixmap(QPixmap.fromImage(showImage))
            self.ui.Video_Label.setScaledContents(True)  # 让图片自适应 label 大小

            # 状态栏显示信息
            self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_t, total_t = self.calculate_time(self.current_f, self.total_f, self.fps)
            self.ui.LoadingInfo.setText("文件名：{}        {}({})".format( self.file, current_t, total_t))

    def calculate_time(self, c_f, t_f, fps):
        total_seconds = int(t_f / fps)
        current_sec = int(c_f / fps)
        c_time = "{}:{}:{}".format(int(current_sec / 3600), int((current_sec % 3600) / 60), int(current_sec % 60))
        t_time = "{}:{}:{}".format(int(total_seconds / 3600), int((total_seconds % 3600) / 60), int(total_seconds % 60))
        return c_time, t_time

    def show_pic_back(self):
        # 获取视频当前帧所在的帧数
        self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 设置下一次帧为当前帧-2
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_f-2)
        # 读取一帧
        success, frame = self.cap.read()
        if success:
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.ui.Video_Label.setPixmap(QPixmap.fromImage(showImage))

            # 状态栏显示信息
            current_t, total_t = self.calculate_time(self.current_f-1, self.total_f, self.fps)
            self.ui.LoadingInfo.setText("文件名：{}        {}({})".format( self.file, current_t, total_t))
        # 快退

    def replay(self):
        self.v_timer = QTimer()  # self
        self.ui.play.setText("暂停")
        # 读取视频
        self.cap = cv2.VideoCapture(self.file)
        if not self.cap:
            print("打开视频失败")
            return
        # 获取视频FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # 获得码率
        # 获取视频总帧数
        self.total_f = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 获取视频当前帧所在的帧数
        self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 设置定时器周期，单位毫秒
        self.v_timer.start(int(1000 / self.fps))
        print("FPS:".format(self.fps))
        # 连接定时器周期溢出的槽函数，用于显示一帧视频
        self.v_timer.timeout.connect(self.show_pic)
        # 连接按钮和对应槽函数，lambda表达式用于传参
        print("play OK")

    def last_img(self, t):
        self.ui.play.setText("暂停")
        if t:
            # 断开槽连接
            self.v_timer.timeout.disconnect(self.show_pic)
            # 连接槽连接
            self.v_timer.timeout.connect(self.show_pic_back)
            self.v_timer.start(int(1000 / self.fps) / 2)
        else:
            self.v_timer.timeout.disconnect(self.show_pic_back)
            self.v_timer.timeout.connect(self.show_pic)
            self.v_timer.start(int(1000 / self.fps))
        # 快进

    def next_img(self, t):
        self.ui.play.setText("暂停")
        if t:
            self.v_timer.start(int(1000 / self.fps) / 2)  # 快进
        else:
            self.v_timer.start(int(1000 / self.fps))
# 暂停播放

    def go_pause(self):
        if  self.ui.play.text() == "暂停":
            self.v_timer.stop()
            self.ui.play.setText("播放")
        elif self.ui.play.text() == "播放":
            self.v_timer.start(int(1000/self.fps))
            self.ui.play.setText("暂停")
