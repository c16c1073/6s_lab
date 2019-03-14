import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gray = cv2.imread( 'img/mackey.JPG',0 )
w,h=gray.shape
print(w,h)
sw,sh=int(w/5),int(h/5)
print(sw,sh)
s_gray=cv2.resize(gray,(sw,sh))

plt.subplot(121)
plt.imshow( s_gray,cmap=plt.cm.gray )
plt.axis('off')
#plt.show()


#re_gray=s

xx,yy=np.mgrid[ 0:s_gray.shape[0],0:s_gray.shape[1] ]
#print( xx.shape  )
#print(yy)

plt.subplot(122)
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(xx,yy,s_gray,rstride=1,cstride=1,cmap=plt.cm.gray)
ax.view_init(80,80)
#plt.show()


#------------------------------horizonal differential filter-------------------------------------

fig,[ax1,ax2,ax3]=plt.subplots( 1,3,figsize=(15,3) )
fig.suptitle( 'hdf',fontsize=20 )

#zennsinnsabunnkinnzi
kernel=np.array([ [0,0,0],[0,-1,1],[0,0,0] ])
filtered=cv2.filter2D( s_gray,-1,kernel )
ax1.imshow( filtered,cmap=plt.cm.gray )
# koutaisabunnkinnzi
kernel=np.array( [ [0,0,0],[-1,1,0],[0,0,0] ] )
filtered=cv2.filter2D( s_gray,-1,kernel )
ax2.imshow( filtered,cmap=plt.cm.gray )
# tyuusinnsebunnkinniz
kernel=np.array( [ [0,0,0],[-1,0,1,],[0,0,0,] ] )
filtered=cv2.filter2D( s_gray,-1,kernel )
ax3.imshow( filtered,cmap=plt.cm.gray )

#cv2.imwirte('bibun.jpg',filtered)

#超解像 東工大　奥富　

plt.show()
#--------------------焦点ボケの画像だとやっぱりエッジ検出不可っぽい
