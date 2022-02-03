# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:09:01 2022

@author: csnadmin

This python code aiming for reading convergence data from a list of excel files
corresponding to data from different sections. 
The code returns data in dictionary format and datum in pandas serie format

Function sort_date sorts the date in ascending and then use the sorted index to 
sort data so that the data and date are both sorted.
"""
import pandas as pd
from datetime import datetime
import numpy as np
from collections import defaultdict
import glob
import pylab as pl
import dufte
import os
import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt


pl.style.use(dufte.style)
cm = 1/2.54


work_dir = './Exempelfall'
sectionName = os.path.basename(work_dir)

def DataReader(work_dir):
    files = glob.glob(work_dir+'/'+'*.xlsx')
    data = defaultdict(list)
    #datum = defaultdict(list)
    datum =pd.Series(dtype='int32')
    
    for i, file in enumerate(files):

        data_org = pd.read_excel(file)
        data_arr = np.array(data_org.iloc[:,2:5])
        data_arr = data_arr[:,[1,0,2]]
        data[i].append(data_arr)
        #datum[i].append(data_org.iloc[0,1])
        
        ### when more than 1 section have measured on the same date, they are
        ### converted to have exactly the same date and stored in a panda serie.
        ### The index of the datum is used to control which section
        if len(str(data_org.iloc[0,1])) > 6:
            datum = datum.append(pd.Series(str(data_org.iloc[0,1])[0:6]))
        else:
            datum = datum.append(pd.Series(str(data_org.iloc[0,1])))
        datum = datum.reset_index(drop=True)
    return data,datum

data,datum = DataReader(work_dir)


def sort_date(data,datum):
    new_datum = pd.to_datetime(datum,format='%y%m%d') ##convert string to date
    new_datum = new_datum.sort_values(axis=0)   #sort date in acending order  
    new_data = defaultdict(list)
    for i in np.arange(len(new_datum)):
        ind = new_datum.index[i] # new index for sorting the data
        new_data[i].append(data[ind][0]) #create a new dict according to the sorted date index
    return new_data,new_datum

data,datum =sort_date(data,datum)

# Räkna efter hur många punkter det är i mätningen. 
NoPoints = data[0][0].shape[0]

# Ta ut minstakvadratplanet för varje mätning
# def GenerateLocalSystem(data)


ones=np.array([[1,1,1,1,1]])
dataLocal = []
CGglobal  = []
Eigglobal = []
DistCG    = []

for i in data:
    A= np.array(data[i])
    A= np.asmatrix(A)
    CG = np.matmul(ones,A)/5.0
    A= A-CG
    dataLocal.append(A)
    CGglobal.append(CG)
    M=np.dot(A.T,A)
    U, s, vh = np.linalg.svd(M)
    Eigglobal.append(U[:,2])
    foo = np.sqrt(np.matmul(np.square(A),[1,1,1]))
    DistCG.append(foo)

# return DistCG, Eigglobal

# skapa det lokala koordinatsystemet

eZt = np.array([0,0,1])
eX = []
eY = []
eZ = []

for i in data:
    eYt = np.cross(eZt,Eigglobal[i].T)
    eYt = eYt/np.linalg.norm(eYt)
    eZt = np.cross(Eigglobal[i].T,eYt)
    eZt = eZt/np.linalg.norm(eZt)
    eXt = np.cross(eYt,eZt)
    eXt = eXt/np.linalg.norm(eXt)
    eX.append(eXt)
    eY.append(eYt)
    eZ.append(eZt)
# Paketera ihop koordinatvektorerna och returena dem (todo)




# plot för att kolla att alla koordinattransformationer fungerat
def plotPointsAndRotation(dataLocal):
    ax = a3.Axes3D(pl.figure(),auto_add_to_figure=False)
    fig=pl.gcf()
    fig.add_axes(ax) 
    A = dataLocal[0]
    B = np.matmul(A,[eX[0].T,eY[0].T,eZ[0].T]).T
    
    ax.scatter(A[:,0],A[:,1],A[:,2], label ="Inmätta punkter")
    ax.scatter(B[:,0],B[:,1],B[:,2], label ="punkter i roterat plan")   
    vec = Eigglobal[0]
    x,y,z =np.concatenate(vec) 
    ax.quiver(0,0,0,x,y,z,length=1)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_box_aspect([1,1,1])
    pl.savefig(sectionName + "Pointplacements.pdf", format="pdf") 

#Plottar en kontrollplot för koordinattransformen, kommentera bort om du inte vill ha den.
#plotPointsAndRotation(dataLocal)

# Räkna om DistCG i det lokala koordinatsystemet (Bör vara väldigt liten skillnad mellan systemen om mätningarna är bra) Y och Z är de nya koordinaterna. X är out of plane och stryks i detta fall.

DistCGplane = []
ones=np.array([[1,1]])

for i in range(len(dataLocal)):
    A = dataLocal[i]
    B = np.matmul(A,[eX[i].T,eY[i].T,eZ[i].T]).T
    B = B[:,1:3]
    B = np.multiply(B,B)
    B = np.matmul(B,ones.T)
    B = np.sqrt(B)
    DistCGplane.append(B)
    

# Räkna fram förändringen av avståndet mellan varje mätpunkt och tyngdpunkten
PointDist=np.zeros([NoPoints,len(DistCGplane)])
for i in range(len(DistCGplane)):
    tmpDist=DistCGplane[i]-DistCGplane[0]
    PointDist[:,i]=np.asarray(tmpDist[:,0].T)

    
    
    
# Plottfunktionen
# def PlotCurves(datum,PointList)
# kolla antalet punkter, välj rätt lista med labels

#Label. Om du har ett annat antal punkter ärn 5, eller redovisar punkterna i annan ordning, justera listan nedan.
label5 = ["Punkt 1, vägg", "Punkt 2, anfang", "Punkt 3, tak", "Punkt 4, anfang","Punkt 5, vägg" ]

    
fig = pl.figure(constrained_layout=False, figsize=(30*cm, 18*cm),dpi=100)    
#ax= fig.gca()    
for i in range(len(PointDist[:,0])):
    pl.plot(datum,PointDist[i,:],label=label5[i])

pl.title(sectionName)    
dufte.legend()

pl.gcf().autofmt_xdate()
plt.tight_layout()
pl.savefig(sectionName + ".pdf", format="pdf")     
pl.show()