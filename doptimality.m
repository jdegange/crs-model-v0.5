nfactors = 3;
nruns = 18;
nlvls = [10 7 3];
[dCE,X] = cordexch(nfactors,nruns,'linear','tries',50, ...
                   'levels',nlvls,'categorical',[1 2 3]);

% plot design
figure(1);
plot3(dCE(:,1),dCE(:,2),dCE(:,3),'or','markersize',12,'markerfacecolor','r'); grid on;
xlabel('attr1'); ylabel('attr2'); zlabel('attr3');
set(gca,'xtick',1:nlvls(1),'ytick',1:nlvls(2),'ztick',1:nlvls(3));
title(['d-optimal design with ' num2str(nruns) ' runs']);
add_lines(dCE);
axis image;
view(75,20);

% plot histograms
figure(2);
[amin amax] = deal(min(dCE(:)),max(dCE(:)));
subplot(3,1,1); h = histogram(dCE(:,1),1:nlvls(1)); set(gca,'xlim',[amin amax]+[-1 1]);
h.BinWidth = .3;
xlabel('attr1 level'); set(gca,'xtick',1:nlvls(1));
title('Occurrences of each level')

subplot(3,1,2); h = histogram(dCE(:,2),1:nlvls(2)); set(gca,'xlim',[amin amax]+[-1 1]);
xlabel('attr2 level'); ylabel('occurrences'); set(gca,'xtick',1:nlvls(2));
h.BinWidth = .3

subplot(3,1,3); h = histogram(dCE(:,3),1:nlvls(3)); set(gca,'xlim',[amin amax]+[-1 1]);
xlabel('attr3 level'); set(gca,'xtick',1:nlvls(3));
h.BinWidth = .3