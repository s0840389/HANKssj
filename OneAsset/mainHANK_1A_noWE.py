import os
import pickle

mainpath='/home/jamie/OneDrive/Documents/research/HANK/HANK_ssj_git/OneAsset'

os.chdir(mainpath)

import numpy as np
import matplotlib.pyplot as plt

from sequence_jacobian import simple, solved, combine, create_model  # functions
from sequence_jacobian import grids, hetblocks                       # modules

from hh_prob_1A_noWE import *

from sticky_exp import *


# household prob (from package)

def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, pi_e, Pi = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA)
    return e_grid, pi_e, Pi, a_grid


def transfers(pi_e, Div,Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter 
    tax_rule, div_rule = e_grid, e_grid
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T

def wages(w, e_grid,N):
    we = w * e_grid*N
    return we

#hh = hh_prob_1A.hh

hh_ext = hh.add_hetinputs([make_grid, transfers, wages])

@simple
def LS(frisch,vphi,w):
    N=(w/vphi)**frisch
    return N


@simple
def LSss(w,frisch,N):
    vphi=w/N**(1/frisch)
    return vphi

# firms

@simple
def firm(w,N, Z, pi, mu, kappa):
    Y=N*Z
    L = Y / Z
    Div = Y - w * L - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return L, Div,Y


# monetary and fiscal

@simple
def monetary(pi, rstar, phi):
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r


#@simple
#def fiscal(r, B):
#    Tax = r * B
#    return Tax


@simple
def fiscalSS(B,r,G):
    Tax=(r*B+G)    
    return Tax

@solved(unknowns={'B': 4.,'Tax': 0.3 }, targets=['B_res','tax_res'], solver="broyden_custom")
def fiscal(Tax,B,r, G,gammatax,rhotax,taxss,Bss):
    B_res=B(-1)*(1+r)+G-Tax-B
    tax_res = taxss*(Tax(-1)/taxss)**rhotax*(B(-1)/Bss)**(gammatax*(1-rhotax))-Tax
    return B_res,tax_res


# market clearing

@simple
def mkt_clearing(A, C, Y, B, pi, mu, kappa,G,L,N):
    asset_mkt = A - B
    #labour_mk=L-N
    goods_mkt = Y - C - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y -G
    return asset_mkt, goods_mkt#,labour_mk

# philips curve
@simple
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))\
               - (1 + pi).apply(np.log)
    return nkpc_res

# steady state

@simple
def nkpc_ss(Z, mu):
    w = Z / mu
    return w

blocks_ss = [hh_ext, firm, monetary, fiscalSS, mkt_clearing, nkpc_ss,LSss]

hank_ss = create_model(blocks_ss, name="One-Asset HANK SS")

calibration = {'eis': 0.70, 'frisch': 0.5, 'rho_e': 0.966, 'sd_e': 0.92, 'nE': 7,
               'amin': 0.0, 'amax': 150, 'nA': 250, 'N': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi': 1.5, 'B': 5.6,'cbar': 0.0,'G':0.2,
               'rhotax':0.5,'gammatax':0.6}

unknowns_ss = {'beta': 0.986}
targets_ss = {'asset_mkt': 0}

ss = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")

ss['Bss']=ss['B']
ss['taxss']=ss['Tax']

# dynamic model

blocks = [hh_ext, firm, monetary, fiscal, mkt_clearing, nkpc,LS]
hank = create_model(blocks, name="One-Asset HANK")

T = 300

exogenous = ['rstar', 'Z']
unknowns = ['pi', 'w']
targets = ['nkpc_res', 'asset_mkt']

J_ha = hh_ext.jacobian(ss, inputs=['Tax','Div', 'r','w','N'], T=T)

G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T,Js={'hh': J_ha})


J_ha_sticky=stick_jacob(J_ha,0.94) # Reduce forwarding lookingness of hh Jacobian and resolve model

G_sticky = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T,Js={'hh': J_ha_sticky})

# Consumption decomposition chart

drstar = -0.0025 * 0.8 ** (np.arange(T)[:, np.newaxis])

drstar_fwd=np.roll(drstar,10)
drstar_fwd[:10]=0


dc_lin=100*G['C']['rstar'] @ drstar/ss['C']

fig,ax =plt.subplots()

tt=np.arange(0,T)

yyoldminus=0*tt[0:24]
yyoldplus=0*yyoldminus

bcolor=['darkblue','darkgreen','grey','gold','gold']
iter=0
for i in ['r','Div','Tax','w','N']:
    
    yy=J_ha['C'][i]@G[i]['rstar']@drstar/ss['C']*100 # combine HH jacobian with GE inputs 

    ax.bar(tt[:24],yy[:24,-1].clip(min=0),bottom=yyoldplus,label=i,color=bcolor[iter])
    ax.bar(tt[:24],yy[:24,-1].clip(max=0),bottom=yyoldminus,color=bcolor[iter])
    
    yyoldplus=yy[:24,-1].clip(min=0)+yyoldplus
    yyoldminus=yy[:24,-1].clip(max=0)+yyoldminus

    iter=iter+1

plt.plot(dc_lin[:24], label='Total', linestyle='-', linewidth=2.5)

ax.legend()
plt.title('Decomposition of consumption response to monetary policy (FI)')
plt.show()

fig.savefig('Consump_decomp_noWE.png')

# Output IRFs comparison

varpick='Y'
shockpick='rstar'
shockname='MP shock'

dy_lin=100*G[varpick][shockpick] @ drstar/ss['Y']
dy_lin_sticky=100*G_sticky[varpick][shockpick] @ drstar/ss['Y']
dy_lin_fwd=100*G[varpick][shockpick] @ drstar_fwd/ss['Y']

fig2,ax=plt.subplots()

plt.plot(dy_lin[:24], label='linear', linestyle='-', linewidth=2.5)
#plt.plot(dy_lin_fwd[:24], label='linear forward', linestyle='--', linewidth=2.5)
plt.plot(dy_lin_sticky[:24], label='Linear sticky', linestyle='--', linewidth=2.5)

tit=varpick+' response to 1% '+shockname

plt.title(tit)
plt.xlabel('quarters')
plt.ylabel('% deviation from ss')
plt.legend()
plt.show()



Adist=np.sum(ss.internals['hh']['D'],axis=0)
agrid=ss.internals['hh']['a_grid']

fig4, ax=plt.subplots()

ax.bar(agrid,Adist)
plt.xlim([0,100])
plt.title('Iliquid asset distribution')
plt.show()

fig4.savefig('asset_dist_noWE.png')

#plt.plot(ss.internals['hh']['a_grid'],ss.internals['hh']['c'].swapaxes(0,1))



