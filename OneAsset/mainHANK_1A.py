
# ces between production on skill

import os
import pickle

mainpath='/home/jamie/OneDrive/Documents/research/HANK/HANK_ssj_git/OneAsset'

os.chdir(mainpath)

import numpy as np
import matplotlib.pyplot as plt

from sequence_jacobian import simple, solved, combine, create_model  # functions
from sequence_jacobian import grids, hetblocks                       # modules

from hh_prob_1A import *

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

def wages(w, e_grid):
    we = w * e_grid
    return we

hh1 = hh.add_hetinputs([make_grid, transfers, wages])

#hh0=hetblocks.hh_labor.hh

#hh1 = hh0.add_hetinputs([make_grid, transfers, wages])

#hh = hh_prob_1A.hh

def compute_consumption(c):

    c01 = np.zeros_like(c)
    c01_10 = np.zeros_like(c)
    c10_35 =np.zeros_like(c)
    c35_65 =np.zeros_like(c)
    c65_90 = np.zeros_like(c)
    c90_99 = np.zeros_like(c)
    c99 = np.zeros_like(c)
    c01[0,:] = c[0,:]
    c01_10[1,:] = c[1,:]
    c10_35[2,:] = c[2,:]
    c35_65[3,:] = c[3,:]
    c65_90[4,:] = c[4,:]
    c90_99[5,:] = c[5,:]
    c99[6,:] = c[6,:]
    return c01, c01_10, c10_35, c35_65, c65_90, c90_99, c99


def compute_labor(ne):
    n01 = np.zeros_like(ne)
    n01_10 =np.zeros_like(ne)
    n10_35 =np.zeros_like(ne)
    n35_65 =np.zeros_like(ne)
    n65_90 = np.zeros_like(ne)
    n90_99 = np.zeros_like(ne)
    n99 = np.zeros_like(ne)
    n01[0,:] = ne[0,:]
    n01_10[1,:] = ne[1,:]
    n10_35[2,:] =ne[2,:]
    n35_65[3,:] =ne[3,:]
    n65_90[4,:] = ne[4,:]
    n90_99[5,:] = ne[5,:]
    n99[6,:] = ne[6,:]

    ncheck=np.zeros_like(ne)
    #ncheck[0,:]=1
    ncheck=ne*1

    return n01, n01_10, n10_35, n35_65, n65_90, n90_99, n99, ncheck


def labor_supply(n, e_grid):
    ne = e_grid[:, np.newaxis] * n
    return ne

hh_ext = hh1.add_hetoutputs([labor_supply,compute_consumption,compute_labor])


# firms

@simple
def firm(Y, w, Z, pi, mu, kappa):
    L = Y / Z
    Div = Y - w * L - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return L, Div


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
def mkt_clearing(A, NE, C, L, Y, B, pi, mu, kappa,G):
    asset_mkt = A - B
    labor_mkt = NE - L
    goods_mkt = Y - C - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y -G
    return asset_mkt, labor_mkt, goods_mkt

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

blocks_ss = [hh_ext, firm, monetary, fiscalSS, mkt_clearing, nkpc_ss]

hank_ss = create_model(blocks_ss, name="One-Asset HANK SS")

calibration = {'eis': 0.5, 'frisch': 0.5, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 7,
               'amin': 0.0, 'amax': 150, 'nA': 250, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi': 1.5, 'B': 5.6,
               'cbar': 0.0,'G':0.0,'rhotax':0.9,'gammatax':0.9}

unknowns_ss = {'beta': 0.986, 'vphi': 0.8}
targets_ss = {'asset_mkt': 0, 'labor_mkt': 0}

ss = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")

ss['Bss']=ss['B']
ss['taxss']=ss['Tax']

# dynamic model

blocks = [hh_ext, firm, monetary, fiscalSS, mkt_clearing, nkpc]
hank = create_model(blocks, name="One-Asset HANK")

T = 300

exogenous = ['rstar', 'Z']
unknowns = ['pi', 'w', 'Y']
targets = ['nkpc_res', 'asset_mkt', 'labor_mkt']

J_ha = hh_ext.jacobian(ss, inputs=['Tax','Div', 'r','w'], T=T)

G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T,Js={'hh': J_ha})
#G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T)


J_ha_sticky=stick_jacob(J_ha,0.94) # Reduce forwarding lookingness of hh Jacobian and resolve model

G_sticky = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T,Js={'hh': J_ha_sticky})

# Consumption decomposition chart

drstar = -0.0025 * 0.61 ** (np.arange(T)[:, np.newaxis])

drstar_fwd=np.roll(drstar,10)
drstar_fwd[:10]=0


dc_lin=100*G['C']['rstar'] @ drstar/ss['C']

fig,ax =plt.subplots()

tt=np.arange(0,T)

yyoldminus=0*tt[0:24]
yyoldplus=0*yyoldminus

bcolor=['darkblue','darkgreen','grey','gold']

iter=0
for i in ['r','Div','Tax','w']:
    
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

fig.savefig('Consump_decomp.png')

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
plt.title('asset distribution')
plt.show()

fig4.savefig('Asset_dist.png')


#plt.plot(ss.internals['hh']['a_grid'],ss.internals['hh']['c'].swapaxes(0,1))


rho_r, sig_r = 0.61, -0.01/4
rstar_shock_path = {"rstar": sig_r * rho_r ** (np.arange(T))}

#td_nonlin = hank.solve_impulse_nonlinear(ss, unknowns, targets, rstar_shock_path)
td_lin = hank.solve_impulse_linear(ss, unknowns, targets, rstar_shock_path)

zdist=np.sum(ss.internals['hh']['D'],axis=1)

fig5, ax=plt.subplots()

N01IRF=(G['N01']['rstar']@drstar)[0:24]/ss['N01']*100
N50IRF=(G['N35_65']['rstar']@drstar)[0:24]/ss['N35_65']*100
NIRF=(G['N']['rstar']@drstar)[0:24]/ss['N']*100
N50IRFv2=td_lin['N35_65'][0:24]*100/ss['N35_65']
N99IRF=(G['N99']['rstar']@drstar)[0:24]/ss['N99']*100

plt.plot(N01IRF, label='Lowest income', linestyle='-', linewidth=2.5)
plt.plot(N50IRF, label='Middle', linestyle='-', linewidth=2.5)
plt.plot(N99IRF, label='Top', linestyle='-', linewidth=2.5)
plt.plot(N50IRFv2, label='Middle (check)', linestyle='--', linewidth=2.5)
plt.plot(NIRF, label='Aggregate', linestyle='-', linewidth=2.5)

plt.title('Labour response')
plt.legend()

plt.show()

fig5.savefig('NIRF_dist.png')




plt.plot(G['Y']['rstar']@drstar)
plt.show()


xx=['N01']

xx=['N01','N01_10','N10_35','N35_65','N65_90','N90_99','N99']

Nirf=0*td_lin['N']
Nss=0
iter=0
for ii in xx:
    Nirf=Nirf+td_lin[ii]/ss[ii]*100 #*zdist[iter]*100
    Nss=Nss+ss[ii]#*zdist[iter]
    iter=iter+1


fig6, ax=plt.subplots()

plt.plot(Nirf[0:20], label='BU', linestyle='-', linewidth=2.5)
plt.plot(100*td_lin['NE'][0:20], label='TD', linestyle='-', linewidth=2.5)
plt.title('Labour response')
plt.legend()