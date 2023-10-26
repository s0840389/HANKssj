
function [cpol,kpol,Vk]=polupdate(prices,Vk_f,p,grid,sstate) % policy function update

r=prices(1);
w=prices(2);

cminusstar=(Vk_f*p.bbeta).^(1/-p.ssigma);

kminusstar=(grid.kgrid-w*grid.zgrid'+cminusstar)/(1+r); % implied starting point

cpol=zeros(grid.nk,grid.nz);

y=(1+r)*grid.kgrid+w*grid.zgrid'-grid.kgrid(1);

for s=1:grid.nz
alow=kminusstar(1,s);
bcind=grid.kgrid<alow;
h=sortrows([ kminusstar(:,s),cminusstar(:,s)],1);
cpols=griddedInterpolant(h(:,1),h(:,2));
cpol(:,s)=cpols(grid.kgrid).*(1-bcind)+bcind.*y(:,s);
end

%% capital supply

kpol=min((1+r)*grid.kgrid+w*grid.zgrid'-cpol,grid.kgrid(end));


Vk=(cpol.^(-p.ssigma))*sstate.Pz'*(1+r);


end

