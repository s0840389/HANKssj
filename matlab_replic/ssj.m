clear all
close all
load('steady.mat')

tic;

epsh=[sstate.w sstate.w]'*10^-4; % derivative step

T=300; % sequence horizon

nprices=size(sstate.prices,1);


cpol=zeros(grid.nk,grid.nz);
Ppol=sparse(grid.nk*grid.nz,grid.nk*grid.nz);
kpol=zeros(grid.nk,grid.nz,nprices);

Vk_f=zeros(size(cpol)); % just function of prices not distribution

Ds=zeros(grid.nk*grid.nz,1);

FK=zeros(T,T,nprices); % fake news matrix

kl=repmat(grid.kgrid,grid.nz,1);
kdashl=sstate.kpol(:);
sstate.JDr=reshape(sstate.JD,[grid.nk,grid.nz]);


%% Calculate Fake News Matrix

for t=0:(T-1) % solve consumption policies and transition matrices 
    
    tt=T-t;

    for pp =1:nprices
    
            prices=sstate.prices;
    
        if tt==T
            Vk_f(:,:,pp)=sstate.Vk;
            prices(pp)=prices(pp)+epsh(pp);
        end
        
        % go 1 step backwards
        [cpol,kpol,Vk_f(:,:,pp)]=polupdate(prices,Vk_f(:,:,pp),p,grid,sstate); % new policies in response to news
        [Ppol]=tranupdate(kpol,grid,sstate); % Transition matrix in respnse of news
    
        % fill in fakeg news matrix for iniital impact
        s=t+1;
        
        FK(1,s,pp)=sum(sum(sstate.JDr.*(kpol-sstate.kpol)))/epsh(pp); % effect on impact of news shock
        Ds=(sstate.JD'*Ppol)'-sstate.JD;
       
        for ttt=2:T % go forward in time after fake news shocks    
            FK(ttt,s,pp)=sum(Ds.*kdashl)/epsh(pp); %effect on outcome 
            Ds=(Ds'*sstate.Ph)';
        end
    
    end
end


% jacobian of K with respect to prices

JK=zeros(T,T,nprices);

for pp = 1:nprices
    JK(1,:,pp)=FK(1,:,pp);
    JK(:,1,pp)=FK(:,1,pp);
end

for tt=2:T
    for pp=1:nprices
        JK(tt,2:end,pp)=JK(tt-1,1:end-1,pp)+FK(tt,2:end,pp);
    end
end

%% Auclert et al figures

JKr=JK(:,:,1);
FKr=FK(:,:,1);

JKw=JK(:,:,2);
FKw=FK(:,:,2);

figure(1)
clf

subplot(3,1,1)
plot(JKr(:,[1,25,50,75,100]))
legend(string([1,25,50,75,100]))
title('Jacobian for K at diffrent horizon shocks to r')

subplot(3,1,2)
plot(FKr(:,1))
legend(string(1))
title('Fake news matrix for initial shock')

subplot(3,1,3)
plot(FKr(:,[25,50,75,100]))
legend(string([25,50,75,100]))
title('Fake news paths for shocks at different horizons')

toc;

%% Impulse response functions

%build Jacobian 
% F(x,Z)=0=K'-G(x,z)
%x= [r w k]

dwdk=(1-p.aalpha)*(p.aalpha)*sstate.Kd^(p.aalpha-1)*sstate.N^(-p.aalpha);
drdk=p.aalpha*(p.aalpha-1)*sstate.Kd^(p.aalpha-2)*sstate.N^(1-p.aalpha);

drdz=p.aalpha*sstate.Kd^(p.aalpha-1)*sstate.N^(1-p.aalpha);
dwdz=(1-p.aalpha)*sstate.Kd^(p.aalpha)*sstate.N^(-p.aalpha);

Fk=JKr*drdk+JKw*dwdk;
Fk=Fk(2:end,1:end-1)-eye(T-1,T-1);

Fz=JKr*drdz+JKw*dwdz;
Fz=Fz(1:end-1,1:end-1);

dz=1*power(0.90*ones(T-1,1),(0:T-2)');
dx=-1*(Fk\eye(T-1,T-1))*Fz*dz;


figure(2)
clf
plot(dx(1:50))
title('Response of K to a TFP shock')