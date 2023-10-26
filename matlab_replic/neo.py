

import numpy as np

from scipy import linalg

import matplotlib.pyplot as plt

import copy

####################################################################################
# Parameters
####################################################################################

delta=0.025
alpha=0.11
beta=0.981
rhoa=0.90
sigma=1


h=0.0001


####################################################################################
# steady state
####################################################################################

# x=[k,a]
# y=[c,pk]

yss=np.array([0, 0],dtype='float')
xss=np.array([0 ,0],dtype='float')

yss[1]=1/beta+delta

xss[1]=1
xss[0]=(yss[1]/alpha)**(1/(alpha-1))

yss[0]=(xss[0]**alpha)-xss[0]*delta;

xss=xss[:,None]
yss=yss[:,None]

def Fsys(ydash,y,xdash,x):

    F=np.ndarray(shape=(4,1),dtype='float')

    F[0]=y[0]**(-sigma)-beta*(ydash[1]-delta)*ydash[0]**(-sigma)
    F[1]=y[1]-x[1]*alpha*x[0]**(alpha-1)
    F[2]=xdash[0]-x[0]*(1-delta)-(x[1]*x[0]**alpha-y[0])
    F[3]=np.log(xdash[1])-np.log(x[1])*rhoa
    return(F)

Fss=Fsys(yss,yss,xss,xss)

if sum(Fss)>10**-4:
    print('Steady state not zero')


####################################################################################
# Solve dynamics with SGU (F(y',y,x',x,eps)=0)
####################################################################################


# derivatives of system
F1=np.ndarray(shape=(4,2),dtype='float') # dF/dx'
F2=np.ndarray(shape=(4,2),dtype='float') # dF/dy'
F3=np.ndarray(shape=(4,2),dtype='float') # dF/dx
F4=np.ndarray(shape=(4,2),dtype='float') # dF/dy

for i in range(0,2):
    hh=np.ndarray(shape=(2,1),dtype='float')
    hh[:]=0
    hh[i]=h
    F1[:,[i]]=((Fsys(yss,yss,xss+hh,xss)-Fsys(yss,yss,xss,xss)) /h)
    F3[:,[i]]=((Fsys(yss,yss,xss,xss+hh)-Fsys(yss,yss,xss,xss)) /h)


for i in range(0,2):
    hh=np.ndarray(shape=(2,1),dtype='float')
    hh[:]=0
    hh[i]=h
    F2[:,[i]]=((Fsys(yss+hh,yss,xss,xss)-Fsys(yss,yss,xss,xss)) /h)
    F4[:,[i]]=((Fsys(yss,yss+hh,xss,xss)-Fsys(yss,yss,xss,xss)) /h)


# QZ decomposition to solve model
s, t, alphaqz,betaqz, Q, Z = linalg.ordqz(np.concatenate([F1,F2],axis=1), -1*np.concatenate([F3,F4],axis=1),sort='ouc')

relev = abs(s.diagonal())/abs(t.diagonal())

slt   = (relev>=1)*1
nk    = sum(slt) # Number of state Variables based on Eigenvalues

z21=Z[nk:,:nk]
z11=Z[:nk,:nk]
s11=s[:nk,:nk]
t11=t[:nk,:nk]

z11i=linalg.solve(z11,np.eye(nk))

# solved model
# y=ybar+gx*(x-xbar)
# x'=xbar+hx(x-xbar)
gx=np.real(z21@z11i)
hx=np.real(z11 @ linalg.solve(s11,t11) @ z11i)


# TFP IRF from peturbation


T=50 # horizon

eps=np.array([0.0,0.01])
eps=eps[:,None]

yirf=np.ndarray(shape=(2,T),dtype='float')
xirf=np.ndarray(shape=(2,T),dtype='float')


xirf[:,0]=(xss+eps).flatten()

for tt in range(0,T):

    xt=xirf[:,tt][:,None]

    yt=yss + gx @ (xt-xss)

    yirf[:,[tt]]=yt
    
    if tt<=T-2:
        xtdash= xss + hx @ (xt-xss)
        xirf[:,[tt+1]]=xtdash


xirfhat=np.log(xirf/(xss*np.ones(xirf.shape)))

yirfhat=np.log(yirf/(yss*np.ones(yirf.shape)))

# plotted figure
print('(1) Normal peturbation solution')
fig,ax=plt.subplots(2,2)

ax[0,0].plot(range(0,T),yirfhat[0,:])
ax[0,0].set_title('Consumption')

ax[0,1].plot(range(0,T),yirfhat[1,:])
ax[0,1].set_title('pk')

ax[1,1].plot(range(0,T),xirfhat[0,:])
ax[1,1].set_title('capital')

ax[1,0].plot(range(0,T),xirfhat[1,:])
ax[1,0].set_title('tfp')



# truncated sequence space soln using jacobians

T=200

dz=np.power(rhoa*np.ones(T),range(0,T))*0.01
dz=dz[:,None]

Fk=np.ndarray(shape=(T*3,T*3),dtype='float')
Fk[:,:]=0


B=np.concatenate([F1[:,[0]],F4],axis=1)
B=B[0:3,:]

Bdash=np.concatenate([np.zeros([4,1]),F2],axis=1)
Bdash=Bdash[0:3,:]

Bdash2=np.concatenate([F3[:,[0]],np.zeros([4,2])],axis=1)
Bdash2=Bdash2[0:3,:]

for tt in range(0,T):

     Fk[tt*3+0:3*tt+3,tt*3+0:3*tt+3]=B

     if tt not in [0]:
        Fk[(tt-1)*3+0:3*(tt-1)+3,tt*3+0:3*tt+3]=Bdash

     if tt not in [T-1]:
        Fk[(tt+1)*3+0:3*(tt+1)+3,tt*3+0:3*tt+3]=Bdash2

Fz=np.ndarray(shape=(T*3,T),dtype='float')
Fz[:,:]=0

Bz=F3[0:3,[1]]
Bzdash=F1[0:3,[1]]

for tt in range(0,T):

     Fz[tt*3+0:3*tt+3,[tt]]=Bz

     if tt not in [0]:

        Fz[(tt-1)*3+0:3*(tt-1)+3,[tt]]=Bzdash


RFk=np.linalg.matrix_rank(Fk)

if RFk<T*3:
    print('Matrix not invetible, Rank='+str(RFk))
    
Fki=np.linalg.inv(Fk)

#Fki=np.linalg.solve(Fk,np.eye(T*3))

dk=-1*Fki @ Fz @ dz

dkr=dk.reshape(T,3)

ss=np.concatenate((xss[0,None],yss[0:2]))


dkrIRF=np.log((dkr+ss.transpose())/ss.transpose())

# plotted figure (sequence space IRF)
print('(2) Sequence space soln')

fig2,ax=plt.subplots(2,2)

ax[0,0].plot(range(0,50),dkrIRF[0:50,0])
ax[0,0].set_title('Consumption')

ax[0,1].plot(range(0,50),dkrIRF[0:50:,2])
ax[0,1].set_title('pk')

ax[1,0].plot(range(0,50),dz[0:50])
ax[1,0].set_title('tfp')

ax[1,1].plot(range(0,50),dkrIRF[0:50,0])
ax[1,1].set_title('capital')


# fake news algorithm approach

# pk is a X input
# K is Y output
# c is like v

# fake news

cfake=np.ndarray(shape=(T,1),dtype='float')
pkfake=np.ndarray(shape=(T,T),dtype='float')

pkfake=np.ones([T,1])*yss[1]
pkfake[T-1]=yss[1]+0.0001

# policy function reaction 

#for j in range(0,T-1):
#   jT=T-j-2
#   cfake[jT,s]=(beta*cfake[jT+1,s]**(-sigma)*(pkfake[jT+1,s]-delta))**(-1/(sigma))


