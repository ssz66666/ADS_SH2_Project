import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, freqz

fig = plt.figure()
ax = fig.gca( projection='3d')
plt.rcParams.update({'font.size': 22})


df = pd.DataFrame()
for i in range(10):
    # df1 = pd.read_csv('Training-Reformated/'+ str(i+1) + '.csv')
    df1 = pd.read_csv('Constant/Reformatted/' + str(i + 1) + '.csv')
    # df.append(df1, ignore_index = True)
    frames = [df, df1]
    df = pd.concat(frames, ignore_index=True)

df2 = pd.read_csv('Constant/Reformatted/freeliving-pub.csv')

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    z_min, z_max = z.min() - 1, z.max() + 1
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h), np.arange(z_min, z_max, h))
    return xx, yy, zz


def plot_contours(ax, clf, xx, yy, zz, **params):
    X = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    Y = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contour(X, Y, Z, **params)
    return out


x = df['AccX'].values
y = df['AccY'].values
z = df['AccZ'].values
label = df['Activity'].values

xx = df2['AccX'].values
yy = df2['AccY'].values
zz = df2['AccZ'].values
Room = df2['Room'].values

acc = np.array((x, y, z), dtype=float).transpose()
print(label)

C = 100
model = svm.SVC(kernel = 'rbf', gamma = 0.7,  C=C)
model = model.fit(acc, label)

colours = ['b', 'r', 'g', 'k']
labels = ['Sitting', 'Walking', 'Lying', 'Custom']

colour = []
for i in range(0,len(label)):
    c = colours[label[i]-1]
    colour.append(c)

ax = plt.axes(projection = '3d')

ax.scatter3D(x, y, z, c=colour,  marker='o', s= 50)
plt.xlabel('x')
plt.ylabel('y')
#plt.zlabel('z')

# def butter_lowpass(cutoff, fs, order):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=True)
#     return b, a
#
# def butter_lowpass_filter(data, cutoff, fs, order):
#     b, a = butter_lowpass(cutoff, fs, order=order)
#     y = lfilter(b, a, data)
#     return y
#
# order = 4
# fs = 1
# cutoff = 15
#
# filtered = butter_lowpass_filter(acc, cutoff, fs, order)

# print(filtered)
# print(acc)

# X0, X1, X2 = acc[:, 0], acc[:, 1], acc[:, 2]
# xx, yy, zz = np.meshgrid(X0, X1, X2, indexing='ij')

#plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# X = model.predict(np.c_[filtered[:,0].ravel(), filtered[:,1].ravel(), filtered[:,2].ravel()])
X = model.predict(np.c_[x.ravel(), y.ravel(), z.ravel()])
# X = model.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
print(X)
#
# outdf = pd.DataFrame(X)
# outdf.to_csv("experiment.csv")

c = 0
d = 0
e=0
for i in range(len(label)):
    # if Room[i] != Room[i-1] and X[i-1] != 2:
    #     c+=1
    if X[i] == 2:
        if label[i] == X[i]:
            c+=1
        if label[i] != X[i]:
            d+=1
    if label[i] == 2:
        e+=1
    if label[i] == X[i]:
        c+=1
    # if X[i] == 2:
    #     c+=1




# print(X)

lb1= ax.scatter(1.5, 1, c='b', label='sitting')
lb2= ax.scatter(0.3, 0.3, c='r', label='walking')
lb3= ax.scatter(0.7, 0.9, c='g', label='lying')
lb4= ax.scatter(1.2, 0.2, c='k', label='custom')
plt.legend(handles=[lb1,lb2,lb3,lb4],loc=1,title=r'$\bf{Activity}$')
lb1.remove();lb2.remove();lb3.remove();lb4.remove()

print(c/len(label))
# print(e)
# print(d)

plt.show()
