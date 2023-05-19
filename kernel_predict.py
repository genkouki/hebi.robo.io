import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#import seaborn as sns
from itertools import combinations_with_replacement
import csv
from mpl_toolkits.mplot3d import Axes3D 

def kernel(x1i, x1j,x2i, x2j, beta=0.00005):
        
    return np.exp(- beta * np.sum((x1i - x1j)**2+(x2i - x2j)**2))

import csv

#配列宣言
angle2 = []
angle3 = []
position1 = []
position2 = []
angle = []
position=[]

#csvファイルを指定
MyPath = '20221215takamori-test_csv_out.csv'

#csvファイルを読み込み
with open(MyPath) as f:

    reader = csv.reader(f)

    #csvファイルのデータをループ
    for row in reader:

        #D列を配列へ格納
        angle2.append(float(row[3]))
        #E列を配列へ格納
        angle3.append(float(row[4]))


        #H列を配列へ格納
        position1.append(float(row[7]))
        position2.append(float(row[8]))
        
angle.insert(10, angle2)
angle.insert(10, angle3)
position.insert(10, position1)
position.insert(10, position2)

X1 = position1#実際の計測データ
X2 = position2#実際の計測データ
Y1 = angle2#実際の計測データ
Y2 = angle3#実際の計測データ

#print(position1)
#print(position2)
#print(angle2)
#print(angle3)



# ハイパーパラメータ
beta = 0.00005

# データ数
#N = X.shape[0]
N =100

# グラム行列の計算
K = np.zeros((N, N))
for i, j in combinations_with_replacement(range(N), 2):#Nこの要素から2つをえらぶすべての組み合わせ
    K[i][j] = kernel(X1[i], X1[j],X2[i], X2[j])
    K[j][i] = K[i][j]

# 重みを計算
alpha1 = np.linalg.inv(K).dot(Y1)
alpha2 = np.linalg.inv(K).dot(Y2)

#print(alpha1)
#print(angle2)
#print(alpha2)
#print(angle3)


# カーネル回帰
def kernel_predict(X1,x1,X2,x2, alpha, beta):
    Y = 0
    for i in  range(100):
        Y += alpha[i] * kernel(X1[i],x1,X2[i], x2, beta)
    return Y

# 正則化項の係数
lam = 0.5
alpha_r1 = np.linalg.inv(K + lam * np.eye(K.shape[0])).dot(Y1)
alpha_r2 = np.linalg.inv(K + lam * np.eye(K.shape[0])).dot(Y2)

alpha=np.column_stack((alpha_r1,alpha_r2))
np.savetxt("alpha.csv", alpha, delimiter=",")

#alpha_demo=np.loadtxt("alpha.csv",delimiter=",",skiprows=0,usecols=(0,1))

# 回帰によって結果を予測        
X1_axis = np.arange(0, 600, 30)
X2_axis = np.arange(0, 600, 30)
Y1_predict_r = np.zeros((20, 20))
Y2_predict_r = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        Y1_predict_r[i][j] = kernel_predict(X1, X1_axis[i],X2,X2_axis[j], alpha_r1, beta)
        Y2_predict_r[i][j] = kernel_predict(X1, X1_axis[i],X2,X2_axis[j], alpha_r2, beta)
#Y1_predict_1=kernel_predict(X1, 100,X2,300, alpha_r1, beta)
#Y2_predict_2=kernel_predict(X1, 100,X2,300, alpha_r2, beta)
#描画エリアの作成
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')

#軸ラベル
#ax.set_xlabel("X1",size=15,color="black")
#ax.set_ylabel("X2",size=15,color="black")
#ax.set_zlabel("Y",size=15,color="black")

#for i in range(99):
#データのプロット
#ax.plot(X1_axis, X2_axis,Y1_predict_r , "o", color="blue")

#メッシュ作成
#x1,x2  = np.meshgrid(X1, X2)
#X1_axis,X2_axis = np.meshgrid(X1_axis, X2_axis)
#print(np.meshgrid(X1_axis, X2_axis))
#print(X1_axis)
#for i in range(20):
   # for j in range(20): 
        #ax.scatter(X1_axis[i],X2_axis[j] ,Y1_predict_r[i][j], "o", color="green")
        #ax.scatter(X1_axis[i],X2_axis[j] ,Y2_predict_r[i][j], "o", color="black") 

#ax.scatter(X1,X2, Y1, color='red')
#ax.scatter(100,300, Y1_predict_1, color='red')
#ax.scatter(X1,X2,Y2,color='blue')
#ax.scatter(100,300, Y2_predict_2, color='blue')

#ax.plot_wireframe(X2_axis, X1_axis,Y1_predict_r, color='blue',linewidth=1)#メッシュ表示
#ax.plot_wireframe(X1_axis, X2_axis,Y2_predict_r, color='red',linewidth=1)
#ax.plot_surface(X1_axis, X2_axis,Y1_predict_r, cmap='winter', linewidth=0.3)#曲面描画
#ax.plot_surface(X1_axis, X2_axis,Y2_predict_r, cmap='summer', linewidth=0.3)
    
#グラフの回転
#ax.view_init(elev=0, azim=90)


#グラフの表示
#plt.show()