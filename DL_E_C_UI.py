#!/usr/bin/python
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DL_UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import numpy as np
import cv2
import tensorflow.keras.models
import pickle, sklearn
import os, sys
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 780)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 221, 65))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 1, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 1, 1, 1)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 80, 221, 261))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_3 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.listWidget = QtWidgets.QListWidget(self.layoutWidget1)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout.addWidget(self.listWidget)
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(250, 0, 1011, 721))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.graphicsView = QtWidgets.QGraphicsView(self.layoutWidget2)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout_2.addWidget(self.graphicsView)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_5 = QtWidgets.QPushButton(self.layoutWidget2)
        self.pushButton_5.setObjectName("pushButton_5")
        self.horizontalLayout.addWidget(self.pushButton_5)
        self.pushButton_8 = QtWidgets.QPushButton(self.layoutWidget2)
        self.pushButton_8.setObjectName("pushButton_8")
        self.horizontalLayout.addWidget(self.pushButton_8)
        self.spinBox = QtWidgets.QSpinBox(self.layoutWidget2)
        self.spinBox.setMinimumSize(QtCore.QSize(211, 22))
        self.spinBox.setMinimum(-10)
        self.spinBox.setMaximum(10)
        self.spinBox.setSingleStep(1)
        self.spinBox.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.spinBox.setProperty("value", 0)
        self.spinBox.setDisplayIntegerBase(10)
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout.addWidget(self.spinBox)
        self.pushButton_4 = QtWidgets.QPushButton(self.layoutWidget2)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout.addWidget(self.pushButton_4)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 680, 231, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.lcdNumber = QtWidgets.QLCDNumber(self.horizontalLayoutWidget)
        self.lcdNumber.setFrameShape(QtWidgets.QFrame.Box)
        self.lcdNumber.setSmallDecimalPoint(False)
        self.lcdNumber.setDigitCount(2)
        self.lcdNumber.setProperty("intValue", 0)
        self.lcdNumber.setObjectName("lcdNumber")
        self.horizontalLayout_2.addWidget(self.lcdNumber)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(10, 370, 224, 224))
        self.label_6.setObjectName("label_6")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(50, 600, 141, 71))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.pushButton_6 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout_3.addWidget(self.pushButton_6)
        self.pushButton_7 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout_3.addWidget(self.pushButton_7)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionauthor_weber = QtWidgets.QAction(MainWindow)
        self.actionauthor_weber.setObjectName("actionauthor_weber")
        self.menu.addAction(self.actionauthor_weber)
        self.menubar.addAction(self.menu.menuAction())
        self.pushButton.clicked.connect(self.read_model_path)
        self.pushButton_2.clicked.connect(self.loadmodel)
        self.pushButton_3.clicked.connect(self.show_image_path)
        self.listWidget.itemClicked.connect(self.process)
        self.pushButton_4.clicked.connect(self.get_change_num)
        self.pushButton_5.clicked.connect(self.show_zero)
        self.pushButton_6.clicked.connect(self.zoomout)
        self.pushButton_7.clicked.connect(self.zoomin)
        self.pushButton_8.clicked.connect(self.show_123)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "线虫计数"))
        MainWindow.setWindowIcon(QtGui.QIcon('TF32.ico'))
        self.pushButton.setText(_translate("MainWindow", "读取模型"))
        self.label.setText(_translate("MainWindow", "读取Model文件"))
        self.pushButton_2.setText(_translate("MainWindow", "加载模型"))
        self.label_2.setText(_translate("MainWindow", "读取后加载模型"))
        self.pushButton_3.setText(_translate("MainWindow", "打开图片文件夹"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">图片列表</p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">增减数量</p></body></html>"))
        self.pushButton_4.setText(_translate("MainWindow", "确认"))
        self.pushButton_5.setText(_translate("MainWindow", "显示判断为没有线虫或者边缘区域"))
        self.pushButton_8.setText(_translate("MainWindow", "显示判断为线虫的区域"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt;\">线虫条数</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">三条以上判断</p></body></html>"))
        self.pushButton_6.setText(_translate("MainWindow", "放大+"))
        self.pushButton_7.setText(_translate("MainWindow", "缩小-"))
        self.menu.setTitle(_translate("MainWindow", "关于"))
        self.actionauthor_weber.setText(_translate("MainWindow", "作者：Weber"))
    '''相关函数'''

    # 读取model
    def read_model_path(self):
        self.model_file = QtWidgets.QFileDialog.getExistingDirectory(None, '选择模型', '.')

    def loadmodel(self):
        self.model = tensorflow.keras.models.load_model(self.model_file + '/cnn.model')
        self.lb = pickle.loads(open(self.model_file + '/cnn_lb.pickle', "rb").read())
        QtWidgets.QMessageBox.information(None, '提醒', '加载成功', QtWidgets.QMessageBox.Ok)

    def show_image_path(self):
        self.image_path = QtWidgets.QFileDialog.getExistingDirectory(None, '加载图片文件', '.')
        self.img_list = os.listdir(self.image_path)
        for Item in self.img_list:
            if Item.endswith('.jpg') or Item.endswith('.tif'):
                self.listWidget.addItem(Item)
            else:
                continue

    # 图像分割和识别

    def process(self, Index, med_blur_ksize=29, cnany_minVal=20, canny_maxVal=50, mor_ksize=(25, 25)):
        img_filename = self.image_path + '/' + self.listWidget.item(self.listWidget.row(Index)).text()
        self.num = 0
        ori_img = cv2.imread(img_filename)
        img_show = ori_img.copy()
        img_show = cv2.copyMakeBorder(img_show, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img_show_2 = img_show.copy()
        img_info = ori_img.shape
        gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2BGRA)
        med_blur_img = cv2.medianBlur(gray_img, med_blur_ksize)  # 中值滤波处理

        sharp_kernel = np.array([[0, -1, 0], [-1, 5.7, -1], [0, -1, 0]], np.float32)
        sharpen_img = cv2.filter2D(med_blur_img, -1, kernel=sharp_kernel)  # 锐化

        edges_img = cv2.Canny(sharpen_img, cnany_minVal, canny_maxVal)  # Canny边缘
        mor_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, mor_ksize)  # 定义形态学处理椭圆核
        closed_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, mor_kernel)  # 先膨胀后腐蚀的形态学去噪
        dst_img = cv2.adaptiveThreshold(closed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 1)
        boder_img = cv2.copyMakeBorder(dst_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        cnts, _ = cv2.findContours(boder_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # 只需要最外层轮廓
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for cnt in cnts[2:]:
            boder_img = cv2.drawContours(boder_img, [cnt], 0, 255, -1)
        boder_img[:, :1], boder_img[:, -1:], boder_img[:1, :], boder_img[-1:, :] = [0], [0], [0], [0]
        contours, _ = cv2.findContours(boder_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for i in range(0, len(contours)):
            area = cv2.contourArea(contours[i])

            if img_info[0] * img_info[1] * 0.5 > area > img_info[0] * img_info[1] * 0.0002:
                dst = np.zeros((img_info[0], img_info[1], img_info[2]), dtype=np.uint8)
                mask = cv2.drawContours(dst, contours, i, (255, 255, 255), -1)
                img_new = cv2.copyTo(ori_img, mask=mask)
                x, y, w, h = cv2.boundingRect(contours[i])
                img_cropped = img_new[y:y + h, x:x + w]
                image = cv2.resize(img_cropped, (224, 224))

                # scale图像数据
                image = image.astype("float32") / 255.0

                # 对图像进行拉平操作
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

                # 预测
                preds = self.model.predict(image)

                # 得到预测结果以及其对应的标签
                j = preds.argmax(axis=1)[0]
                label = self.lb.classes_[j]
                # 在图像中把结果画出来
                if label == 'none':
                    text = '0:{:.2f}%'.format(preds[0][j] * 100)
                    img_show_2 = cv2.rectangle(img_show_2, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img_show_2, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.num += 0
                elif label == 'one':
                    text = '1:{:.2f}%'.format(preds[0][j] * 100)
                    img_show = cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img_show, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.num += 1
                elif label == 'two':
                    text = '2:{:.2f}%'.format(preds[0][j] * 100)
                    img_show = cv2.rectangle(img_show, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(img_show, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    self.num += 2
                else:
                    # label == 'others':
                    text = '[{}]at least 3:{:.2f}%'.format(str(i), preds[0][j] * 100)
                    img_show = cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(img_show, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    img_cropped = cv2.resize(img_cropped, (224,224))
                    img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
                    frame = QtGui.QImage(img_cropped, 224, 224, QtGui.QImage.Format_RGB888)
                    pix = QtGui.QPixmap.fromImage(frame)
                    self.label_6.setPixmap(pix)
                    judge_num, _ = QtWidgets.QInputDialog.getInt(None, '识别为三条以上人工判断', '由于计算机判断为三条以上，需人工加以判断\n程序左下角图片中线虫数量:', 3, 0, 10, 1)
                    self.num += judge_num


            else:
                continue

        # 绘制外轮廓
        self.label_6.setPixmap(QtGui.QPixmap(""))
        img_show = cv2.drawContours(img_show, cnts[1], -1, (255, 0, 0), 5)
        img_show = cv2.resize(img_show, (1024, 960))
        self.img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
        self.zoomscale = 1
        frame = QtGui.QImage(self.img_show, 1024, 960, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(frame)
        self.item = QtWidgets.QGraphicsPixmapItem(pix)
        self.scene = QtWidgets.QGraphicsScene()
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        img_show_2 = cv2.drawContours(img_show_2, cnts[1], -1, (255, 0, 0), 5)
        img_show_2 = cv2.resize(img_show_2, (1024, 960))
        self.img_show_2 = cv2.cvtColor(img_show_2, cv2.COLOR_BGR2RGB)
        self.lcdNumber.display(self.num)

    def show_zero(self):
        QtWidgets.QMessageBox.information(None, "零条人工判断", "对识别为0条和边缘的区域判断，确定最终数量\n在按钮右侧加减上边缘及误判", QtWidgets.QMessageBox.Ok)
        self.zoomscale = 1
        frame = QtGui.QImage(self.img_show_2, 1024, 960, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(frame)
        self.item = QtWidgets.QGraphicsPixmapItem(pix)
        self.scene = QtWidgets.QGraphicsScene()
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)

    def show_123(self):
        self.zoomscale = 1
        frame = QtGui.QImage(self.img_show, 1024, 960, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(frame)
        self.item = QtWidgets.QGraphicsPixmapItem(pix)
        self.scene = QtWidgets.QGraphicsScene()
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)

    def zoomout(self):
        self.zoomscale += 0.05
        if self.zoomscale >= 2:
            self.zoomscale = 2
        self.item.setScale(self.zoomscale)

    def zoomin(self):
        self.zoomscale -= 0.05
        if self.zoomscale <= 0.2:
            self.zoomscale = 0.2
        self.item.setScale(self.zoomscale)

    def get_change_num(self):
        self.num += self.spinBox.value()
        self.lcdNumber.display(self.num)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
