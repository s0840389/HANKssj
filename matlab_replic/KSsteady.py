import numpy as np

from scipy import linalg, interpolate
from numba import jit
import matplotlib.pyplot as plt
import time
import copy

## Parameters

delta=0.025
alpha=0.11
beta=0.981
rhoa=0.90
sigma=1

nk=250
nz=2

kgrid=np.linspace(0,30,nk)[:,None]

zgrid=np.array([0.6,1])[:,None]

Pz=np.array([[0.8,0.2],[0.05,0.95]])

@jit(nopython=True)
def ergdisc(P): # finding ergodic distribution of markov process

    eP = np.linalg.eig(P.transpose())
    uz=np.real(eP[1])
    uzi=np.where(np.real(eP[0])==1)
    f=uz[:,uzi[0][0]]/sum(uz[:,uzi[0][0]])
    return(f)


zerg=ergdisc(Pz)[:,None]

N=sum(zgrid*zerg)

r0=0.02

Kd=((r0+delta)/(alpha*N**(1-alpha)))**(1/(alpha-1)) # capital demand

w=(1-alpha)*(Kd**alpha)*N**(-alpha) # wage

y=kgrid*(1+r0)+w*zgrid.transpose()

r=copy.copy(r0)

# solve consumption policy

cpol=copy.copy(y)

dff=10

iter=0

while (dff>10**(-6)) &(iter<1000):
    
    iter=iter+1
    cstar=((cpol @ Pz.transpose())**(-sigma)*beta*(1+r))**(-(1/sigma))

    kminus=(kgrid+cstar-w*zgrid.transpose())/(1+r)

    cpolold=copy.copy(cpol)

    for s in range(0,nz):
        kmin=kminus[0,s]
        splinein=np.concatenate([kminus[:,[s]],cstar[:,[s]]],axis=1)
        splinein=splinein[splinein[:,0].argsort()]
        cpolspline=interpolate.CubicSpline(splinein[:,0],splinein[:,1]) # consumption policy    
        bcind=kgrid<kmin # borrowing constrain indicator
        cpol[:,[s]]=cpolspline(kgrid)*(1-bcind)+bcind*y[:,[s]] # consumption policy on grid
    
    dff=max((cpolold-cpol).flatten())

kpol=kgrid*(1+r)+w*zgrid.transpose()-cpol # capital choice

kpol=np.minimum(kpol,kgrid[nk-1]*np.ones([nk,nz]))

# capital transition matrix

klowind=np.ndarray([nk,nz],dtype='int')
khighind=np.ndarray([nk,nz],dtype='int')
klowwgt=np.ndarray([nk,nz],dtype='float')

for s in range(0,nz):

    klowindx=(kpol[:,[s]]>=kgrid.transpose())
    klowind[:,s]=(klowindx*1).sum(axis=1)-1
    khighind[:,[s]]=klowind[:,[s]]+1   

    khighind[:,[s]]=np.minimum(khighind[:,[s]],nk*np.ones([nk,1]))

    klowwgt[:,[s]]=1-(kpol[:,[s]]-kgrid[klowind[:,s]])/(kgrid[khighind[:,s]]-kgrid[klowind[:,s]]+10**-6)