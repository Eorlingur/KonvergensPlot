import pylab as pl
import numpy as np 
import matplotlib as mp

chekAbs=False
#las in forsta filen(nollmatning)
X2 = np.loadtxt(’0.dat’)
#Byt plats pa X och Y
conv = pl.matrix(’0,1,0;1,0,0;0,0,1’) X = X2*conv
#Flytta origo till tyngdpunkten CG
CG=sum(X/5) X=X−CG
x=X[:, 0] y=X[:, 1] z=X[:, 2]


#M blir troghetsmatrisen for matpunkterna
M= np.dot(X.T,X)
#Ta fram minsta egenvektorn for M och #anvand som normalvektor t i l l planet.U, s, V=np.linalg.svd(M)
#for horisontella tunnlar
P=[−U[0, 2]/U[1, 2], 0]
pl.figure(1)
pl.clf()
pl.plot(x, np.polyval(P, x))
pl.plot(x, y, ’ro’)
ax = pl.gca()
ax.set_aspect(’equal’)
pl.title(’Punkterna i horisontalplan med snitt’)

# Vrid till lokalt plan.
N= np.mat([P[0], 1])/np.sqrt(P[0]**2+1) # 3D
N3=U[:, 2].T
#For tunnel 3 med daligt uppsatta prisman #N3[0 ,0]= −1.0
#N3[0 ,1]=0.0 #N3[0 ,2]=0.0
#ta fram nytt koordinatsystem
Xtmp=np.cross(N3, pl.matrix([0,0,1]))
Xtmp=Xtmp/np.dot(Xtmp,Xtmp.T)**0.5
#om möjligt bor X peka i positiv riktning.Sign far ej bli 0 X t m p=X t m p * n p.s i g n( X t m p [ 0 , 0 ] + 1 e − 1 5 )
Ytmp=np.cross(Xtmp,N3)
#kolla att Y ar uppat
Ytmp=Ytmp*np.sign(Ytmp[0,2]+1e−15) #Bygg matrisen med koordinataxlar
U=np.bmat([np.mat(Xtmp.T) ,np.mat(Ytmp.T) ,N3.T])
#projicera i lokalt plan #3D med N3
#projektionen(denna kraver lite eftertanke)
OST=(X.T−N3.T*(N3*X.T)).T*U Res=(OST−X*−U)[: ,2] OST1nk=OST[0 ,:]
OST2nk=OST[1 ,:]
OST3nk=OST[2 ,:]
OST4nk=OST[3 ,:]
OST5nk=OST[4 ,:]
#hitta rotationsriktningen
order =(np.double(OST1nk[0, 0] < OST5nk[0, 0])−.5)*2 order = OST1nk[0 , 0] < OST5nk[0 , 0]
Pos1 = 4*order+6*(1−order)
Pos2 = 1*order+3*(1−order)

Pos3 = 2
Pos4 = 3*order+1*(1−order) Pos5 = 6*order+4*(1−order) PosCG = 5
#
pl.figure(2)
pl.clf()
pl.plot(OST1nk[0, 0], OST1nk[0, 1], 
	pl.plot(OST2nk[0, 0], OST2nk[0, 1], ’ro’) 
	pl.plot(OST3nk[0, 0], OST3nk[0, 1], ’ro’)
	pl.plot(OST4nk[0, 0], OST4nk[0, 1], ’ro’)
	pl.plot(OST5nk[0, 0], OST5nk[0, 1], ’ro’)
	pl.plot(0,0, ’ko’)
#
ax = pl.gca()
ax.set_aspect( ’equal’ )
pl.title( ’Punkterna i lokaltplan ’)

#las in datumfilen och anvand som skala
datum = pl.csv2rec(’tider.txt’, names = ’date’)
dist = len(datum.date)


#Las in resten av
OST1nk = OST1nk.T
OST2nk = OST2nk.T
OST3nk = OST3nk.T
OST4nk = OST4nk.T
OST5nk = OST5nk.T

for i in range(1,len(datum.date)): #antalet datum ger antalet filer
	fname = ’%d.txt’% i
	Y = np.loadtxt(fname)
	Y = np.mat(Y*conv)−CG #byter plats pa y och x och flyttar origo t m p=Y.T
	tmp=tmp−(N3.T*N3*tmp) #projicerar ner i planet
	filerna

	tmp=tmp.T*U #byter till planets koordinater 
	Res=np.bmat([Res,(tmp−Y*−U)[: ,2]]) 
	OST1nk = np.bmat([OST1nk, tmp[0 ,:].T]) #staplar ihop
	OST2nk = np.bmat([OST2nk, tmp[1 ,:].T]) 
	OST3nk = np.bmat([OST3nk, tmp[2 ,:].T]) 
	OST4nk = np.bmat([OST4nk, tmp[3 ,:].T]) 
	OST5nk = np.bmat([OST5nk, tmp[4 ,:].T]) #slut pa linalgen.

last = len(OST1nk.T)−1

#berakna punkternas medelavstand t i l l CG
CGx =(OST1nk[0 , :]+OST2nk[0 , :] +OST3nk[0 , :]+OST4nk[0 , :]+OST5nk[0 ,:])/5
CGy =(OST1nk[1 , :]+OST2nk[1 , :] +OST3nk[1 , :]+OST4nk[1 , :]+OST5nk[1 ,:])/5

tmpcg = np.bmat(’CGx;CGy’)
fooz = np.zeros((1, len(datum))) 
tmpcg = np.bmat( ’tmpcg; fooz’)
 
#koordinaterna relativt CG
o1nc =(OST1nk−tmpcg)[0:2, :].A#fimpar z ifall att..o2nc =(OST2nk−tmpcg)[0:2, :].A
o3nc =(OST3nk−tmpcg)[0:2, :].A
o4nc =(OST4nk−tmpcg)[0:2, :].A
o5nc =(OST5nk−tmpcg)[0:2, :].A
#test av djupfel
depthE=(Res.T−Res[: ,0].T).std(1).A.reshape(dist) 
#depthStd=(Res.T−Res[: ,0].T).std(1).A.reshape(dist)

#plotta rorelserna relativt CG
def plotMovement( corrPos ) :
	"""" Plottar rorelsediagrammet"""
	
	pl.plot(corrPos[0, :] , corrPos[1, :] , ’b’) 
	pl.plot([corrPos[0, 0]], [corrPos[1, 0]], ’ro’) 
	pl.plot([corrPos[0, last]], [corrPos[1, last]], ’ko’) 
	pl.scatter(corrPos[0, :] , corrPos[1, :] ,
		marker=’o’, alpha=0.6, s=np.abs(depthE))

fig =pl.figure(3)
pl.clf()

fig.suptitle(’ Point movements in plane. Red point=start , Black point=last measurement ’ , fontsize=12)
pl.subplot(2, 3, Pos1) plotMovement( o1nc )

if chekAbs:
	plotMovement(OST1nk [ 0 : 2 , : ].A)
pl.title(u’point1’) ax = pl.gca()
ax.set_aspect( ’ equal ’ )
pl.subplot(2, 3, Pos2) plotMovement( o2nc )
if chekAbs:
	plotMovement(OST2nk [ 0 : 2 , : ].A)
pl.title(u’point2’) ax = pl.gca()
ax.set_aspect( ’ equal ’ )
pl.subplot(2, 3, Pos3) plotMovement( o3nc )
if chekAbs:
	plotMovement(OST3nk [ 0 : 2 , : ].A)
pl.title(u’point3’) ax = pl.gca()
ax.set_aspect( ’ equal ’ )
pl.subplot(2, 3, Pos4) plotMovement( o4nc )
if chekAbs:
	plotMovement(OST4nk [ 0 : 2 , : ].A)
pl.title(u’point4’) ax = pl.gca()
ax.set_aspect( ’ equal ’ )
pl.subplot(2, 3, Pos5) plotMovement( o5nc )
if chekAbs:
	plotMovement(OST5nk [ 0 : 2 , : ].A)
pl.title(u’point5’) ax = pl.gca()
ax.set_aspect( ’ equal ’ )
pl.subplot(2 , 3 , PosCG) plotMovement(np.bmat( ’CGx;CGy’ ax = pl.gca()
ax.set_aspect( ’ equal ’ )



pl.title(u ’CGmovement’ )
#ta fram distanserna , pythagoras..
dst1 = np.sqrt(np.add.reduce(o1nc * o1nc)) 
dst2 = np.sqrt(np.add.reduce(o2nc * o2nc)) 
dst3 = np.sqrt(np.add.reduce(o3nc * o3nc)) 
dst4 = np.sqrt(np.add.reduce(o4nc * o4nc)) 
dst5 = np.sqrt(np.add.reduce(o5nc * o5nc)) 
avg =(dst1 + dst2 + dst3 + dst4 + dst5)/5 
avg = avg−avg[0]

#shift for a running average of 5, influences plotting
avgShift = mp.mlab.movavg(avg , 5) 
avgdst3 = mp.mlab.movavg( dst3 , 5)

fig = pl.figure(4) 
pl.clf() 
pl.subplot(4, 1, 1)
pl.errorbar(datum.date, avg,yerr=depthE, fmt=’’) 
line1 = pl.plot(datum.date, avg,’b’)
line2 = pl.plot(datum.date[4:dist], avgShift, ’r’) pl.legend(( line1 , line2 ) ,
	(’measuredconvergence’,’5pt average’), loc=’best’)

#fig.autofmt_xdate()
pl.title(u’average convergence ’)
pl.grid()
pl.subplot(4, 1, 2)
pl.errorbar(datum.date, dst3 − dst3[0],yerr=depthE, fmt=’’) line1 = pl.plot(datum.date, dst3 − dst3[0],’b’)
line2 = pl.plot(datum.date[4:dist], avgdst3 − dst3[0], ’r’) pl.legend(( line1 , line2 ) ,
(’measuredconvergence’,’5ptaverage’), loc=’best’)
#fig.autofmt_xdate()
pl.grid()
pl.title(u’center roof convergence ’)


lowWallC=np.abs(o1nc[0,:]−o5nc[0 ,:]) 
lowWallC=lowWallC−lowWallC[0]
pl.subplot(4, 1, 3)
pl.errorbar(datum.date, lowWallC/2,yerr=depthE, fmt=’’)
line1 = pl.plot(datum.date,
	lowWallC/2, ’b’ , label=’measured convergence ’)
pl.legend( loc=’best’)
fig.autofmt_xdate()
pl.grid()
pl.title(u’lowerwallconvergence ’) pl.draw()
pl.subplot(4, 1, 4)
line1 = pl.plot(datum.date,((Res−Res[:,0]).mean(0).T),’b’) 
line2 = pl.plot(datum.date,(Res−Res[:,0]).std(0).T,’r’)
pl.legend(( line1 , line2 ) ,
	(’average movement’,’movement std’), loc=’best’)
fig.autofmt_xdate()
pl.grid()
pl.title(u ’ depthmovement ’ ) 
pl.draw()