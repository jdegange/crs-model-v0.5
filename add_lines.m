function add_lines(dCE)

% add lines
[xmin xmax ymin ymax zmin zmax] = deal(min(dCE(:,1)),max(dCE(:,1)),...
                                       min(dCE(:,2)),max(dCE(:,2)),...
                                       min(dCE(:,3)),max(dCE(:,3)));
[x y z] = deal(xmin:xmax,ymin:ymax,zmin:zmax);
                                   
                                   
[yy zz] = meshgrid(y,z);
[yy zz] = deal(yy(:),zz(:));
for i = 1:numel(yy), line([xmin xmax],yy(i)*[1 1],zz(:)*[1 1],'color',.4*[1 1 1]); hold on; end

[xx zz] = meshgrid(x,z);
[xx zz] = deal(xx(:),zz(:));
for i = 1:numel(xx), line(xx(i)*[1 1],[ymin ymax],zz(:)*[1 1],'color',.4*[1 1 1]); hold on; end

[xx yy] = meshgrid(x,y);
[xx yy] = deal(xx(:),yy(:));
for i = 1:numel(xx), line(xx(i)*[1 1],yy(:)*[1 1],[zmin zmax],'color',.4*[1 1 1]); hold on; end

hold off;
    


