import os
import cv2
import numpy as np

path = 'origin1'
newpath = 'pre_processed'

def graph_pre_processing(img_filename, med_blur_ksize=29, cnany_minVal=20, canny_maxVal=50, mor_ksize=(25, 25) ):
    ori_img = cv2.imread(img_filename)
    img_info = ori_img.shape
    gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2BGRA)
    med_blur_img = cv2.medianBlur(gray_img, med_blur_ksize)  # 中值滤波处理

    sharp_kernel = np.array([[0, -1, 0], [-1, 5.7, -1], [0, -1, 0]], np.float32)
    sharpen_img = cv2.filter2D(med_blur_img, -1, kernel=sharp_kernel)  # 锐化

    edges_img = cv2.Canny(sharpen_img, cnany_minVal, canny_maxVal)  # Canny边缘
    mor_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, mor_ksize)  # 定义形态学处理椭圆核
    closed_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, mor_kernel)  # 先膨胀后腐蚀的形态学去噪
    dst_img = cv2.adaptiveThreshold(closed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3,1)
    boder_img = cv2.copyMakeBorder(dst_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    cnts, _ = cv2.findContours(boder_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 只需要最外层轮廓
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for cnt in cnts[2:]:
        boder_img = cv2.drawContours(boder_img, [cnt], 0, 255, -1)
    boder_img[:, :1], boder_img[:, -1:], boder_img[:1, :], boder_img[-1:, :] = [0], [0], [0], [0]
    contours, _ = cv2.findContours(boder_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)







    for i  in range(0, len(contours)):

        dst = np.zeros((img_info[0], img_info[1], img_info[2]), dtype=np.uint8)
        mask = cv2.drawContours(dst, contours, i, (255, 255, 255), -1)
        name = newpath + '/' + file.replace('.jpg', '') + str(i) + '.jpg'
        x, y, w, h = cv2.boundingRect(contours[i])
        area = cv2.contourArea(contours[i])

        if img_info[0] * img_info[1] * 0.5 > area > img_info[0] * img_info[1] * 0.0005:
            img_new = cv2.copyTo(ori_img, mask=mask)
            img_cropped = img_new[y:y + h, x:x + w]
            img_resize = cv2.resize(img_cropped, (200, 200))
            cv2.imwrite(name, img_resize)
        else:
            continue



    return dst_img


files =  os.listdir(path)
for i, file in enumerate(files) :
    if file.startswith('img') and file.endswith('.jpg') :
        img_filename = os.path.join(path,file)
        dst_img = graph_pre_processing(img_filename)
        cv2.imshow(img_filename, dst_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()