
function [Ppol]=tranupdate(kpol,grid,sstate) % policy function update


kindl=zeros(grid.nk,grid.nz);
kindh=zeros(grid.nk,grid.nz);

for s=1:grid.nz
kindl(:,s)=sum(kpol(:,s)>=grid.kgrid',2);
end

kindh=min(kindl+1,grid.nk);

kindlwgt=1-(kpol-grid.kgrid(kindl))./(grid.kgrid(kindh)-grid.kgrid(kindl)+0.00001);
kindhwgt=1-kindlwgt;

%transition matrix

rowindP=zeros(grid.nk*grid.nz*2*grid.nz,1);
colindP=zeros(grid.nk*grid.nz*2*grid.nz,1);
valuesP=zeros(grid.nk*grid.nz*2*grid.nz,1);

addii=grid.nk*grid.nz;
stind=1;

Pzl=kron(sstate.Pz,ones(grid.nk,1));

for zi=0:(grid.nz-1)

rowindP(stind:stind+addii-1,1)=(1:grid.nk*grid.nz)';
colindP(stind:stind+addii-1,1)=grid.nk*zi+reshape(kindl(:,:),[],1);
valuesP(stind:stind+addii-1,1)=Pzl(:,zi+1).*reshape(kindlwgt(:,:,1),[],1);
stind=stind+addii;

rowindP(stind:stind+addii-1,1)=(1:grid.nk*grid.nz)';
colindP(stind:stind+addii-1,1)=grid.nk*zi+reshape(kindh(:,:),[],1);
valuesP(stind:stind+addii-1,1)=Pzl(:,zi+1).*reshape(kindhwgt(:,:),[],1);
stind=stind+addii;

end

Ppol=sparse(rowindP,colindP,valuesP,grid.nk*grid.nz,grid.nk*grid.nz);

end