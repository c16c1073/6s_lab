import numpy as np
import cv2

img=cv2.imread( 'img/IMG_2055.JPG' )

"""
cv2.imshow( 'window',img )
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

w,h,ch=img.shape
print('w,h,ch=',w,h,ch)


sh,sw=int(w/5),int(h/5)
part_img=cv2.resize( img,(sw,sh) )
"""
cv2.imshow( 'window',part_img )
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


gray=cv2.cvtColor( part_img, cv2.COLOR_BGR2GRAY )

cv2.imshow( 'window',gray )
cv2.waitKey(0)
cv2.destroyAllWindows()


#<two dimensional FT by numpy>
from matplotlib import pyplot as plt
#------------グレイスケール画像を二次元フーリエ変換して振幅スペクトルを表示(白い部分は低周波成分)

f=np.fft.fft2(gray)
fshift=np.fft.fftshift(f)
magnitude_spectrum=20*np.log( np.abs(fshift) )

plt.subplot(1,1,1),plt.imshow(magnitude_spectrum,cmap='gray')
plt.show()

#print( f )

#---------------------------------二次元フーリエ逆変換 by numpy
rows,cols=gray.shape
crow,ccol=int(rows/2),int(cols/2)
fshift[ crow-30:crow+30, ccol-30:ccol+30 ]=0
f_ishift=np.fft.ifftshift( fshift )
img_back=np.fft.ifft2( f_ishift )
img_back=np.abs(img_back)

plt.subplot(111)
plt.imshow(img, cmap='gray')
plt.show()
plt.subplot(111)
plt.imshow( img_back,cmap='gray' )
plt.show()
plt.subplot(111)
plt.imshow(img_back)
plt.show()







