from PySide2.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Signal, QObject
import torch
import time
from Module.play_video import ShowResultWin as play_video
from Module import seg_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# 让torch判断是否使用GPU

class MySignals(QObject):
    # 定义一种信号，参数是str，即文件的地址
    ms = Signal(str)

model_path_ms = MySignals()
data_path_ms = MySignals()
result_path_ms = MySignals()


class MainWin():  # 主窗口
    def __init__(self):
        super(MainWin, self).__init__()
        # 动态加载主窗口ui界面
        self.ui = QUiLoader().load('ui/main_win.ui')
        self.ui.load_model_btn.clicked.connect(self.load_model)
        self.ui.start_divide_btn.clicked.connect(self.start_divide)
        self.ui.show_result_btn.clicked.connect(self.show_result)
        model_path_ms.ms.connect(self.get_model_path)
        data_path_ms.ms.connect(self.get_data_path)
        result_path_ms.ms.connect(self.get_result_path)

    def load_model(self):
        self.load_model_win = LoadModelWin()
        self.load_model_win.ui.show()

    def get_model_path(self, path):
        text = '成功加载模型！\n模型路径：' + path
        self.ui.load_model_text.setPlainText(text)

    def start_divide(self):
        model_path = self.ui.load_model_text.toPlainText().split('：')[1]
        self.start_divide_win = StartDivideWin(model_path)
        self.start_divide_win.ui.show()

    def get_data_path(self, path):
        text = '成功加载数据！\n数据路径：' + path
        self.ui.data_path_text.setPlainText(text)

    def get_result_path(self, path):
        text = '成功分割数据！\n结果路径：' + path
        self.resultpath= path ##方便传给下面 这里还有点问题 默认是result/test.avi
        self.ui.result_path_text.setPlainText(text)

    def show_result(self):
        self.show_result_win = play_video(self.resultpath)
        self.show_result_win.ui.show()


class LoadModelWin():  # 加载模型的窗口
    def __init__(self):
        super().__init__()
        self.ui = QUiLoader().load('ui/load_model_win.ui')
        self.ui.browse_btn.clicked.connect(self.get_model_path)
        self.ui.ok_btn.clicked.connect(self.ok)
        self.ui.cancel_btn.clicked.connect(self.cancel)

    def get_model_path(self):
        FileDialog = QFileDialog(self.ui.browse_btn)  # 实例化
        FileDialog.setFileMode(QFileDialog.AnyFile)  # 可以打开任何文件
        model_file, _ = FileDialog.getOpenFileName(self.ui.browse_btn, 'open file', 'model/',
                                                   'model files (*.pth *.pt)')
        # 改变Text里面的文字
        self.ui.model_path_text.setPlainText(model_file)

    def ok(self):
        self.ui.close()
        model_path = self.ui.model_path_text.toPlainText()
        model_path_ms.ms.emit(model_path)

    def cancel(self):
        self.ui.close()
        QMessageBox.warning(self.ui.cancel_btn, "警告", "加载模型失败！", QMessageBox.Yes)


class StartDivideWin():  # 开始分割的窗口
    def __init__(self, path):
        super().__init__()
        self.model_path = path
        self.ui = QUiLoader().load('ui/start_divide_win.ui')
        self.ui.browse_btn_1.clicked.connect(self.get_data_path)
        self.ui.browse_btn_2.clicked.connect(self.get_result_path)
        self.ui.ok_btn.clicked.connect(self.ok)
        self.ui.cancel_btn.clicked.connect(self.cancel)

    def get_data_path(self):
        FileDialog = QFileDialog(self.ui.browse_btn_1)
        FileDialog.setFileMode(QFileDialog.AnyFile)
        type = self.ui.type_combo.currentText()
        data_file = ''
        if type=='image':
            data_file = QFileDialog.getExistingDirectory(self.ui.browse_btn_1, "选择图片文件夹", "./")
        elif type=='video':
            data_file = FileDialog.getOpenFileName(self.ui.browse_btn_1, '打开视频文件', './','video files (*.avi *.mp4)')
        elif type=='nii':
            data_file = FileDialog.getOpenFileName(self.ui.browse_btn_1, 'nii文件', './', 'video files (*.avi *.mp4)')
        self.ui.data_path_text.setPlainText(data_file)

    def get_result_path(self):
        FileDialog = QFileDialog(self.ui.browse_btn_2)
        FileDialog.setFileMode(QFileDialog.AnyFile)
        result_file = QFileDialog.getExistingDirectory(self.ui.browse_btn_2, "选择保存结果文件夹", "./")
        self.ui.result_path_text.setPlainText(result_file)

    def ok(self):
        seg = "binary"
        type = self.ui.type_combo.currentText()
        self.ui.close()
        data_path = self.ui.data_path_text.toPlainText()
        data_path_ms.ms.emit(data_path)
        result_path = self.ui.result_path_text.toPlainText()##保存的路径 不包含文件名
        aviname= time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        result_path_ms.ms.emit(result_path+'/'+aviname+'.avi')##传送给主窗口展示的是包含文件名的
        self.ok_win = DividedWin()
        self.ok_win.ui.show()
        seg_image.main("binary", type, data_path, result_path, aviname)



    def cancel(self):
        self.ui.close()
        QMessageBox.warning(self.ui.cancel_btn, "警告", "未能成功开始分割！", QMessageBox.Yes)


class DividedWin():  #
    def __init__(self):
        super().__init__()
        self.ui = QUiLoader().load('ui/divided_win.ui')
        self.ui.ok_btn.clicked.connect(self.ok)
        self.ui.cancel_btn.clicked.connect(self.cancel)

    def ok(self):
        self.ui.close()

    def cancel(self):
        self.ui.close()
        QMessageBox.warning(self.ui.cancel_btn, "警告", "分割结果地址未得到！", QMessageBox.Yes)





def main():
    app = QApplication([])
    start = MainWin()
    start.ui.show()
    app.exec_()


if __name__ == '__main__':
    main()
