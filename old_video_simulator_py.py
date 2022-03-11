#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#シンプル、高速にするためにprocessing版より正確さは無い

#マルチスレッド用
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_NUM_THREADS"] = "8"

import math
import numpy as np

from scipy import signal

import cv2

def dither_matrix(n:int):
    if n == 1:
        return np.array([[0]])
    else:
        first = (n ** 2) * dither_matrix(int(n/2))
        second = (n ** 2) * dither_matrix(int(n/2)) + 2
        third = (n ** 2) * dither_matrix(int(n/2)) + 3
        fourth = (n ** 2) * dither_matrix(int(n/2)) + 1
        first_col = np.concatenate((first, third), axis=0)
        second_col = np.concatenate((second, fourth), axis=0)
        return (1/n**2) * np.concatenate((first_col, second_col), axis=1)

def ordered_dithering(image):
    w = 640
    h = 480

    matrix = dither_matrix(16) #16x16 256段階
    #print(matrix)
    matrix = np.tile(matrix, (round(h / 16), round(w / 16)))
    image[:, :, 0] += matrix
    image[:, :, 1] += matrix
    image[:, :, 2] += matrix
    return image

def RC_filter_kernel(kernel_taps, cutoff, input = [1.0]) :
    kernel = np.array(np.zeros(kernel_taps))
    kernel_acc = input[0]
    for i in range(0, kernel_taps) :
        filter_in = 0
        if i < len(input):
            filter_in = input[i]
        kernel_acc += (filter_in - kernel_acc) * cutoff
        kernel[i] = kernel_acc
    kernel /= np.sum(kernel)
    return kernel

def kernel_delay(kernel, delay):
    return np.append(np.zeros(delay), kernel);

def main():
    print("input image loading", end = " ")
    image_path = "SD color bars.png"
    input_image = cv2.imread(image_path)
    print("done")

    np.set_printoptions(edgeitems=6)

    YUV_U_mask = np.arange(640 * 480).reshape(480, 640).astype(np.int8)
    YUV_U_mask += ((np.arange(640 * 480) / 640 / 2).reshape(480, 640)).astype(np.int8) % 2 * 2
    YUV_U_mask_tmp = YUV_U_mask.copy()
    YUV_U_mask = (((YUV_U_mask_tmp + 0) / 2 % 2) * 2 - 1) * ((YUV_U_mask + 1) % 2)
    YUV_V_mask = (((YUV_U_mask_tmp + 1) / 2 % 2) * 2 - 1) * ((YUV_U_mask + 3) % 2)

    #print(YUV_U_mask)
    #print(YUV_V_mask)

    #input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV) #BGRをYUVに変換
    input_image = input_image.astype(np.float16)

    input_image[:, :, :] = ((input_image[:, :, :] / 255) ** 2.2) * 255;#デガンマ

    #RGB to YIQ
    tmp_input_image = input_image.copy()
    input_image[:, :, 0] = \
        tmp_input_image[:, :, 0] * 0.299 +\
        tmp_input_image[:, :, 1] * 0.587 +\
        tmp_input_image[:, :, 2] * 0.114
    input_image[:, :, 1] = \
        tmp_input_image[:, :, 0] * 0.5959 +\
        tmp_input_image[:, :, 1] * -0.2746 +\
        tmp_input_image[:, :, 2] * -0.3213
    input_image[:, :, 2] = \
        tmp_input_image[:, :, 0] * 0.2115 +\
        tmp_input_image[:, :, 1] * -0.5227 +\
        tmp_input_image[:, :, 2] * 0.3112

    print(input_image.shape)
    input_image[:, :, 0 : 3] /= 255 #YIQを0 ~ 1に変換

    Y_kernel_fir = [
        0.000000000000000e+00 ,
        1.017284023038182e-03 ,
        7.388980554520580e-03 ,
        -4.812812111403899e-18 ,
        -3.784133643203285e-02 ,
        -4.311618446165449e-02 ,
        7.984900500414663e-02 ,
        2.912086905778467e-01 ,
        4.000000000000000e-01 ,
        2.912086905778467e-01 ,
        7.984900500414663e-02 ,
        -4.311618446165449e-02 ,
        -3.784133643203285e-02 ,
        -4.812812111403899e-18 ,
        7.388980554520580e-03 ,
        1.017284023038182e-03 ,
        0.000000000000000e+00 ,]
    Y_kernel = RC_filter_kernel(input = [1], kernel_taps = 32, cutoff = 0.5)
    Y_kernel = kernel_delay(Y_kernel, int(len(Y_kernel)))#カーネルを中心寄せから左寄せにする
    #print(np.floor(Y_kernel * 1000))

    UV_kernel_fir = [
        -1.559268733007751e-18 ,
        -2.278365406039326e-02 ,
        1.052506394780231e-17 ,
        2.754237148973142e-01 ,
        5.000000000000000e-01 ,
        2.754237148973142e-01 ,
        1.052506394780231e-17 ,
        -2.278365406039326e-02 ,
        -1.559268733007751e-18 ,]
    UV_kernel = RC_filter_kernel(input = [1], kernel_taps = 32, cutoff = 0.6)
    UV_kernel = kernel_delay(UV_kernel, int(len(UV_kernel)))

    input_image[:, :, 0] = signal.convolve2d(input_image[:, :, 0], [Y_kernel_fir], boundary='symm', mode='same')#Yに畳み込み
    input_image[:, :, 0] = signal.convolve2d(input_image[:, :, 0], [Y_kernel], boundary='symm', mode='same')#Yに畳み込み

    input_image[:, :, 1] = signal.convolve2d(input_image[:, :, 1], [UV_kernel_fir], boundary='symm', mode='same')#Uに畳み込み
    input_image[:, :, 2] = signal.convolve2d(input_image[:, :, 2], [UV_kernel_fir], boundary='symm', mode='same')#Vに畳み込み
    input_image[:, :, 1] = signal.convolve2d(input_image[:, :, 1], [UV_kernel], boundary='symm', mode='same')#Uに畳み込み
    input_image[:, :, 2] = signal.convolve2d(input_image[:, :, 2], [UV_kernel], boundary='symm', mode='same')#Vに畳み込み
    #input_image[:, :, 0 : 1] = input_image[:, :, 0 : 1].reshape(480, 640).reshape(480, 640, 1)

    #輝度は入力のYそのまま
    Y = np.zeros(640 * 480).reshape(480, 640)
    Y += input_image[:, :, 0]

    #色はU,Vとsin,cosを掛けたもの
    C = np.zeros(640 * 480).reshape(480, 640)
    C += input_image[:, :, 1] * YUV_U_mask[:, :]
    C += input_image[:, :, 2] * YUV_V_mask[:, :]
    C *= 1 #色の電圧　適当

    #色の帯域幅BPF
    C = signal.convolve2d(C, [[-2.049317117073449e-17 ,-8.110584530350413e-02 ,3.357518678133085e-17 ,1.377361887929844e-01 ,-3.291763932625380e-17 ,-1.828053361337005e-01 ,0.000000000000000e+00 ,2.000000000000000e-01 ,0.000000000000000e+00 ,-1.828053361337005e-01 ,-3.291763932625380e-17 ,1.377361887929844e-01 ,3.357518678133085e-17 ,-8.110584530350413e-02 ,-2.049317117073449e-17 ,]], boundary='symm', mode='same')

    #コンポジット映像はただYとCを足しただけ
    YC = Y + C;

    #コンポジットからYC分離
    #Y = signal.convolve(YC, [[0.25, 0.25, 0.25, 0.25]], mode='same')
    YC_delay1H = signal.convolve2d(YC, [[0],[0],[0],[0],[1]], boundary='symm', mode='same')

    YC_fade = abs(signal.convolve2d(YC, [[1.0, 0.0, 0.0, 0.0, -1.0]], boundary='symm', mode='same')) - abs(signal.convolve2d(YC, [[0],[0],[1],[0],[-1]], boundary='symm', mode='same'))
    YC_fade *= 8
    YC_fade = np.clip(YC_fade, -1 , 1)
    YC_fade += 1
    YC_fade /= 2
    YC_fade = signal.convolve2d(YC_fade, [[0.25, 0.25, 0.25, 0.25]], boundary='symm', mode='same')

    vert_Y = signal.convolve2d(YC, [[0.25], [0.0], [0.5], [0.0], [0.25]], boundary='symm', mode='same')
    hori_Y = signal.convolve2d(YC, [[0.25, 0.0, 0.5, 0.0, 0.25]], boundary='symm', mode='same')

    vert_C = -signal.convolve2d(YC, [[0.25], [0.0], [-0.5], [0.0], [0.25]], boundary='symm', mode='same')
    hori_C = -signal.convolve2d(YC, [[0.25, 0.0, -0.5, 0.0, 0.25]], boundary='symm', mode='same')

    Y = np.zeros(640 * 480).reshape(480, 640)
    Y += hori_Y * (1 - YC_fade)
    Y += vert_Y * (YC_fade)

    #Y = YC_fade

    C = np.zeros(640 * 480).reshape(480, 640)
    C += hori_C * (1 - YC_fade)
    C += vert_C * (YC_fade)
    C /= 1 / 2


    output_image = np.zeros(640 * 480 * 3).reshape(480, 640, 3)

    #Y,CからYUVのデコード
    #output_image[:, :, 0] = signal.convolve(Y[::], [[0.5, 0, 0.5]], mode='same')
    output_image[:, :, 0] = Y[::]

    #HPFを使ったシャープ処理
    sharpen_kernel = RC_filter_kernel(input = [1], kernel_taps = 32, cutoff = 0.9)
    sharpen_kernel = kernel_delay(sharpen_kernel, int(len(sharpen_kernel)))

    sharpen = output_image[:, :, 0] - signal.convolve2d(output_image[:, :, 0], [sharpen_kernel], boundary='symm', mode='same') #HPF
    sharpen = signal.convolve2d(sharpen, [sharpen_kernel], boundary='symm', mode='same') - sharpen #HPF
    """
    sharpen = signal.convolve2d(sharpen, [[-9.262467357981924e-03 ,9.759487239003213e-18 ,1.118630335989736e-01 ,0.000000000000000e+00 ,7.999999999999999e-01 ,0.000000000000000e+00 ,1.118630335989736e-01 ,9.759487239003213e-18 ,-9.262467357981924e-03 ,]], boundary='symm', mode='same') #色信号を低減するノッチフィルタ
    """
    output_image[:, :, 0] += sharpen * 0.8
    output_image[:, :, 0] = np.clip(output_image[:, :, 0], 0, 1);

    #UVの処理
    output_image[:, :, 1] = C * YUV_U_mask[::] #Uのデコード
    output_image[:, :, 2] = C * YUV_V_mask[::] #Vのデコード

    output_image[:, :, 1] = signal.convolve2d(output_image[:, :, 1], [[0.25, 0.25, 0.25, 0.25]], boundary='symm', mode='same') * 1 #Uに畳み込み
    output_image[:, :, 2] = signal.convolve2d(output_image[:, :, 2], [[0.25, 0.25, 0.25, 0.25]], boundary='symm', mode='same') * 1 #Vに畳み込み

    output_image[:, :, 1] = signal.convolve2d(output_image[:, :, 1], [[2.944958038392145e-19, -8.670717849467505e-05, -3.319550967880867e-04, -6.269783635101152e-04, -6.891897353101802e-04, 5.067623382275192e-19, 2.194265176660287e-03, 6.770228567408323e-03, 1.449504765874971e-02, 2.572136877875045e-02, 4.010704565915763e-02, 5.647463951586574e-02, 7.289735515160392e-02, 8.702886530667454e-02, 9.660811335016214e-02, 9.999999999999999e-02, 9.660811335016214e-02, 8.702886530667454e-02, 7.289735515160392e-02, 5.647463951586574e-02, 4.010704565915763e-02, 2.572136877875045e-02, 1.449504765874971e-02, 6.770228567408323e-03, 2.194265176660287e-03, 5.067623382275192e-19, -6.891897353101802e-04, -6.269783635101152e-04, -3.319550967880867e-04, -8.670717849467505e-05, 2.944958038392145e-19, ]], boundary='symm', mode='same') * 1 #Uに畳み込み
    output_image[:, :, 2] = signal.convolve2d(output_image[:, :, 2], [[2.944958038392145e-19, -8.670717849467505e-05, -3.319550967880867e-04, -6.269783635101152e-04, -6.891897353101802e-04, 5.067623382275192e-19, 2.194265176660287e-03, 6.770228567408323e-03, 1.449504765874971e-02, 2.572136877875045e-02, 4.010704565915763e-02, 5.647463951586574e-02, 7.289735515160392e-02, 8.702886530667454e-02, 9.660811335016214e-02, 9.999999999999999e-02, 9.660811335016214e-02, 8.702886530667454e-02, 7.289735515160392e-02, 5.647463951586574e-02, 4.010704565915763e-02, 2.572136877875045e-02, 1.449504765874971e-02, 6.770228567408323e-03, 2.194265176660287e-03, 5.067623382275192e-19, -6.891897353101802e-04, -6.269783635101152e-04, -3.319550967880867e-04, -8.670717849467505e-05, 2.944958038392145e-19,
  ]], boundary='symm', mode='same') * 1 #Vに畳み込み
    """
    output_image[:, :, 0] = YC
    output_image[:, :, 1] = 0
    output_image[:, :, 2] = 0
    """

    #output_image[:, :, 1 : 3] /= 2   #UVを-0.5 ~ +0.5に変換
    #output_image[:, :, 1 : 3] += 128 / 255 #UVを0 ~ 1に変換
    output_image[:, :, 0 : 3] *= 255 #YUVを0 ~ 255に変換

    #YIQ to RGB
    tmp_output_image = output_image.copy()
    output_image[:, :, 0] = \
        tmp_output_image[:, :, 0] +\
        tmp_output_image[:, :, 1] * 0.956 +\
        tmp_output_image[:, :, 2] * 0.619
    output_image[:, :, 1] = \
        tmp_output_image[:, :, 0] +\
        tmp_output_image[:, :, 1] * -0.272 +\
        tmp_output_image[:, :, 2] * -0.647
    output_image[:, :, 2] = \
        tmp_output_image[:, :, 0] +\
        tmp_output_image[:, :, 1] * -1.106 +\
        tmp_output_image[:, :, 2] * 1.703

    output_image[:, :, :] = (np.abs(output_image[:, :, :] / 255) ** (1.0 / 2.2)) * 255;#ガンマ

    output_image = ordered_dithering(output_image)

    output_image = np.clip(output_image, 0, 255);
    print("output image saving", end = " ")
    output_image = output_image.astype(np.uint8)
    #output_image = cv2.cvtColor(output_image, cv2.COLOR_YUV2BGR) #YUVをBGRに変換

    cv2.imwrite('output.png', output_image)
    print("done")

if __name__ == "__main__":
    for i in range(1):
        main()
