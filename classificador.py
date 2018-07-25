print(__doc__)
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.io as sio
from sklearn import svm
import os
import fnmatch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans


def make_meshgrid(x, y, h=0.001):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



def set_class(x):
    if x == 'G1':
        return 1
    elif ('S' in x and 'G1' in x):
        return 2
    elif x == 'S':
        return 4
    elif ('S' in x and 'G2' in x):
        return 3
    elif x == 'G2':
        return 5

    else:
        return 0

def set_color(x):
    if x == 'G1':
        return 'red'
    elif ('S' in x and 'G1' in x):
        return 'yellow'
    elif x == 'S':
        return 'blue'
    elif ('S' in x and 'G2' in x):
            return 'orange'
    elif x == 'G2':
        return 'green'
    else:
        return 'black'


class Cell:
    def __init__(self, totalintensity, area, cellcycle):
        self.totalintensity = int(totalintensity)
        self.area = int(area)
        self.cellcycle = str(cellcycle)
        self.Class = set_class(str(cellcycle))





#----------------------------------------START----------------------------------
cells = []
count = 0
join=1
count_g1=0
count_g2=0

#--------------------------------LEGEND-----------------------------------------

G1_patch = mpatches.Patch(color='red', label='G1')
SG1_patch = mpatches.Patch(color='yellow', label='S/G1')
S_patch = mpatches.Patch(color='blue', label='S')
SG2_patch = mpatches.Patch(color='orange', label='S/G2')
G2_patch = mpatches.Patch(color='green', label='G2')
Error_patch = mpatches.Patch(color='black', label='Invalid Data')

#dir = os.path.dirname(os.path.realpath(__file__))
dir = os.getcwd()
for roots, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith('.mat'):
            path = os.path.realpath(os.path.join(roots,file))
            print(path)
            data = (sio.loadmat(path,struct_as_record=True))['storage']
            for case in data:
                cells.append(Cell(case['TotalIntensity'], case['Area'], case['CellCycle'][0][0]))
            count += 1

print (count," files found")


plt.ioff()

n_figures=1
titles=['Area vs Intensity-All']

for i in range(1,n_figures+1):
    plt.figure(i)
    plt.ylabel('Area')
    plt.xlabel('Intensity')
    plt.title(titles[i-1])
    plt.grid(True)
    plt.legend(handles=[G1_patch,SG1_patch,S_patch,SG2_patch,G2_patch,Error_patch])

count=0
for cell in cells:
    color=set_color(cell.cellcycle)
    plt.plot(cell.totalintensity, cell.area, marker = 'o', markersize=3, color=color)
    if color != 'white':
        count += 1

print (count," cells drawn")
plt.savefig('dots.png')
plt.close()


#--------------------------------CLASSIFIER-------------------------------------

dfData = pd.DataFrame(columns=['Intensity', 'Area', 'Class'])
count=0
for cell in cells:
    if cell.Class !=0:
        add=pd.DataFrame({'Intensity':[cell.totalintensity],
                          'Area':[cell.area],
                          'Class':[cell.Class]})
        dfData=dfData.append(add, ignore_index=True)


X = dfData[['Intensity', 'Area']]
X[['Intensity', 'Area']] = preprocessing.scale(dfData[['Intensity', 'Area']])
y = dfData['Class']
y=y.astype('int')
#print(df.to_string())





C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          KMeans(n_clusters=3, random_state=0),
          svm.SVC(kernel='rbf', gamma=1/2, C=C),
          svm.SVC(kernel='poly', degree=1, C=C))
models = (clf.fit(X, y) for clf in models)


# title for the plots
titles = ('SVC with linear kernel',
          'K Means',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')



fig, sub = plt.subplots(3, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)


X0, X1 = X.values[:, 0], X.values[:, 1]
xx, yy = make_meshgrid(X0, X1)



print("Lets print\n")
for clf, title, ax in zip(models, titles, sub.flatten()):
    cf=plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Area')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)


gs = gridspec.GridSpec(3, 1)
ax1 = plt.subplot(gs[2, :]) # row 1, span all columns
cf = ax1.scatter(X0, X1, c=y, cmap=plt.cm.get_cmap('coolwarm',3), s=20, edgecolors='k')
ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim(yy.min(), yy.max())
ax1.set_xlabel("Intensity")
ax1.set_ylabel("Area")
ax1.set_title("No Classificaton")
cbar=plt.colorbar(cf, ax=ax1)
cbar.ax.get_yaxis().set_ticks([])

for j, lab in enumerate(['G1','G1/S','G2/S']):
    cbar.ax.text(1.5, (2 * j + 1) / 6.0, lab, va='center')


print("IM DONE\n")
plt.show()
"""
plt.pause(1)
plt.figure(2)
plt.text(0.05, 0.5, 'Click on me to close!', dict(size=30))
plt.draw()

happy=True
while happy != False:
    happy = plt.waitforbuttonpress(-1)

plt.close()
"""
