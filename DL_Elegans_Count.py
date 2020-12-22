# 导入所需工具包
import numpy as np
import cv2
import tensorflow.keras.models
import pickle
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox



# 读取模型和标签


file = ''
def read_model_path ():
    global file
    file = tk.filedialog.askdirectory(title = '模型')
    return file
def loadmodel ():
    global model, lb
    model = tensorflow.keras.models.load_model(file + '/cnn.model')
    lb = pickle.loads(open(file + '/cnn_lb.pickle', "rb").read())
    lable3.pack()
    botton3.pack()
    return model, lb




# 图像分割和识别

def process (med_blur_ksize=29, cnany_minVal=20, canny_maxVal=50, mor_ksize=(25, 25), num = 0):
    img_filename = tk.filedialog.askopenfilename(title = '选择图片', filetypes = [('图片', '.jpg'),( '图片', '.png'), ('图片', '.jpeg'), ('图片', '.bmp')]) #读取文件地址
    ori_img = cv2.imread(img_filename)
    img_info = ori_img.shape
    gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2BGRA)
    med_blur_img = cv2.medianBlur(gray_img, med_blur_ksize)  # 中值滤波处理

    sharp_kernel = np.array([[0, -1, 0], [-1, 5.7, -1], [0, -1, 0]], np.float32)
    sharpen_img = cv2.filter2D(med_blur_img, -1, kernel=sharp_kernel)  # 锐化

    edges_img = cv2.Canny(sharpen_img, cnany_minVal, canny_maxVal)  # Canny边缘
    mor_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, mor_ksize)  # 定义形态学处理椭圆核
    closed_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, mor_kernel)  # 先膨胀后腐蚀的形态学去噪
    dst_img = cv2.adaptiveThreshold(closed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, hierarchy = cv2.findContours(dst_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 只需要最外层轮廓
# 对图像进行分割
    for i in range(0, len(contours)):
        dst = np.zeros((img_info[0], img_info[1], img_info[2]), dtype=np.uint8)
        mask = cv2.drawContours(dst, contours, i, (255, 255, 255), -1)
        x, y, w, h = cv2.boundingRect(contours[i])
        area = cv2.contourArea(contours[i])
        perimeter = (cv2.arcLength(contours[i], True))
        if area > 1000 and perimeter > 500: #对过小的区域剔除
            img_new = cv2.copyTo(ori_img, mask=mask)
            img_cropped = img_new[y:y + h, x:x + w]
            image = cv2.resize(img_cropped, (224, 224))
            output = image.copy()


            # scale图像数据
            image = image.astype("float32") / 255.0

            # 对图像进行拉平操作
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

            # 预测
            preds = model.predict(image)

            # 得到预测结果以及其对应的标签
            i = preds.argmax(axis=1)[0]
            label = lb.classes_[i]

            if label == 'none':
                num += 0
            elif label == 'one':
                num += 1
            elif label == 'two':
                num += 2
            else:
                # label == 'others':
                cv2.imshow('others', output)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                if tk.messagebox.askyesno(title='图片中有几条',message='图中有3条吗？'):
                    num += 3
                elif tk.messagebox.askyesno(title='图片中有几条',message='图中有4条吗？'):
                    num += 4
                elif tk.messagebox.askyesno(title='图片中有几条',message='图中有5条吗？'):
                    num += 5
                elif tk.messagebox.askyesno(title='图片中有几条',message='图中有6条吗？'):
                    num += 6
                elif tk.messagebox.askyesno(title='图片中有几条',message='图中有7条吗？'):
                    num += 7
                elif tk.messagebox.askyesno(title='图片中有几条',message='纳尼？难道识别错了吗？是2条吗？'):
                    num += 2
                elif tk.messagebox.askyesno(title='图片中有几条',message='不会吧，不会吧。难道是1条'):
                    num += 1
                elif tk.messagebox.askyesno(title='图片中有几条',message='error~是什么都没有吧'):
                    num += 0
                else:
                    num += 8
                cv2.waitKey(1000)
                cv2.destroyAllWindows()




            # 在图像中把结果画出来
            text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
            cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 绘图
            cv2.imshow('predict', output)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()



        else:
            pass
    lable4['text'] = '这张图片中线虫有'+ str(num) + '条\n不包括边缘'
    lable4.pack()





if __name__ == "__main__":
    app = tk.Tk()
    app.title('线虫识别')
    app.geometry('400x280')
    lable1 = tk.Label(app, text='------读取模型和标签------').pack()
    botton1 = tk.Button(app,text = '读取模型',command = read_model_path).pack()
    lable2 = tk.Label(app, text='-----加载模型-----\n-----加载过程中会未响应请等待-----').pack()
    botton2 = tk.Button(app, text='加载模型', command= loadmodel).pack()
    lable3 = tk.Label(app, text='-----加载完成-----\n-----请选择图片-----')
    botton3 = tk.Button(app, text = '选择图片', command = process)
    lable4 = tk.Label(app)
    app.mainloop()

