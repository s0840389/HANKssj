
% parameters
clear all
close all
p.aalpha=0.11;
p.ddelta=0.025;
p.bbeta=0.981;
p.ssigma=1;


% labour productivity
grid.nz=5;

p.rhoz=0.966;

[z,Pz,bounds] = Tauchen(p.rhoz,grid.nz,0.5, 0, 'importance');

grid.zgrid=exp(z)';

sstate.Pz=Pz;

%Pz=[0.8 0.2;
 %   0.05 0.95];

%z=[0.6 1]';

zerg=1/grid.nz*(ones(1,grid.nz)*Pz^100)';

N=zerg'*grid.zgrid;

A=1;

rl=0.00;
rh=0.04;

grid.nk=500;
sstep=10/(grid.nk/2);
grid.kgrid=[linspace(0,10,grid.nk*3/4)';sstep+linspace(10,40,grid.nk*1/4)'];


r=0.02;

excess=10;

iter0=0;

while abs(excess)>10^-6


    iter0=iter0+1;
    
Kd=((p.ddelta+r)/(A*p.aalpha*N^(1-p.aalpha)))^(1/(p.aalpha-1)); % capital demand

w=A*(1-p.aalpha)*Kd^p.aalpha*N^(-p.aalpha); % wage

y=grid.kgrid*(1+r)+w*grid.zgrid'-grid.kgrid(1); % autarky




%% household problem

tol=10e-7;
df=10;
iter=0;

Vk_f=(y.^(-p.ssigma))*Pz'*(1+r); % initial guess on marginal value of capital


while df>tol

    iter=iter+1;
    
    prices=[r w]';

   [cminus,kdash,Vk]=polupdate(prices,Vk_f,p,grid,sstate);


df=norm(Vk-Vk_f,'inf');

Vk_f=Vk;

if iter>999
   display('warning failed household convergence')
    break
end
   

end

%% capital supply

[Ph]=tranupdate(kdash,grid,sstate);

% ergodic distribution 

%JD=zeros(grid.nk*grid.nz,1);
%JD(1)=1;
%JD=(JD'*Ph^300)';

[JDu, JDe]=eigs(Ph');
JDe=real(diag(JDe));
[ve,ie]=min(abs(1-JDe));

JD=real(JDu(:,ie))/sum(real(JDu(:,ie)));

ch=norm(JD'-JD'*Ph,'inf');
if ch>10-6
    display('waring joint dist not converged')
end

% capital supply and r update

Ks=sum(grid.kgrid.*sum(reshape(JD,[grid.nk,grid.nz]),2));

excess=Ks-Kd;

[excess,r,rl,rh]

r0=r;
if excess >0
    r=(r+rl)/2;
    rh=r0;
else
    r=(rh+r)/2;
    rl=r0;
end


if iter0>100
   display('failed to find rate')
    break 
end


end

r=r0;


    figure(1)
clf
plot(grid.kgrid,cminus)

figure(2)
bar(grid.kgrid,sum(reshape(JD,[grid.nk,grid.nz]),2))


% steady state
sstate.cpol=cminus;
sstate.kpol=kdash;
sstate.r=r;
sstate.w=w;
sstate.Y=A*Kd^p.aalpha*N^(1-p.aalpha);
ssstate.A=1;
sstate.N=N;
sstate.pk=r+p.ddelta;
sstate.Ks=Ks;
sstate.Kd=Kd;
sstate.Ph=Ph;
sstate.JD=JD;
sstate.Vk=Vk;
sstate.prices=prices;

save('steady.mat','grid','p','sstate')
