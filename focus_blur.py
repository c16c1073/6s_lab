import numpy as np
import cv2
from matplotlib import pyplot as plt

def load_img():
    img=cv2.imread('img/mackey.JPG')
    return img

def resize(img,w,h):
    re_img=cv2.resize( img,(w,h) )
    return re_img

def show_img(img):
    cv2.imshow( 'window',img )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def transGray(img):
    gray=cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    return gray

def twoDimFFT(gray):# FFT by opencv
    dft=cv2.dft( np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT )
    dft_shift=np.fft.fftshift(dft)# 返り値は複素数型で2ｃｈである
    magnitude_spectrum=20*np.log( cv2.magnitude( dft_shift[:,:,0],dft[:,:,0] ) )
    return magnitude_spectrum

def plt_subplot(data):
    plt.subplot(111)
    plt.imshow( data,cmap='gray' )
    plt.show()
    return 0



if __name__=='__main__':
        img=load_img()
        w,h,ch= list(img.shape)
        W,H=int(w/5),int(h/5)
        print(W,H)
        re_img=resize(img,W,H)
        show_img(re_img)
        gray=transGray(re_img)
        show_img(gray)

        magnitude_spectrum=twoDimFFT(gray)#
        print(magnitude_spectrum)
        plt_subplot(magnitude_spectrum)


    
