%% READ CSV FILE

% open attribute name csv file and store data in attrlines
fstr = 'EntityRiskData_Levels_noCont_simplified_transposed';
fstr = 'attributes_fun';
fid = fopen([fstr '.csv'],'r');
attrlines = textscan(fid,'%s','endofline','\r\n','delimiter','\r\n');
attrlines = attrlines{1};
fclose(fid);

% loop over all rows in csv file
for i = 1:numel(attrlines)-1
    
    % parse row elements (separated by commas) into a cell array
    attrline = strip(attrlines{i+1},',');
    linearray = textscan(attrline,'%s','delimiter',','); linearray = linearray{1};
    
    % store first element of row (attribute key)
    attrKey{i} = linearray{1};
    
    % loop over all remaining columns (attribute levels)
    for j = 1:numel(linearray)-1
        attrVal{i}{j} = linearray{j+1};
    end
end

% count number of levels in each attribute
for i = 1:numel(attrVal)
    nlvls(i) = numel(attrVal{i});
end

%% D-OPTIMAL DESIGN

% coordinate exchange algorithm on N profiles
N = 400;
N = 14;
[dCE,X] = cordexch(numel(nlvls),N,'linear','tries',5, ...
                   'levels',nlvls,'categorical',1:numel(nlvls));

               
%% Save results to file
fname = [fstr '_N=' num2str(N)];
fid = fopen([fname '.csv'],'w+');
for j = 1:numel(attrKey)
    if j~=1, fprintf(fid,'%s',','); end
    fprintf(fid,'%s',attrKey{j});
end
fprintf(fid,'\n');

% loop over profiles
for i = 1:size(dCE,1)
    
    % loop over attributes
    for j = 1:size(dCE,2)
        if j~=1, fprintf(fid,'%s',','); end
        fprintf(fid,'%s',attrVal{j}{dCE(i,j)});
    end
    fprintf(fid,'\n');
    
end
fclose(fid);

%% Plot histograms
for j = 1:size(dCE,2)
    [amin amax] = deal(min(dCE(:)),max(dCE(:)));
    subplot(size(dCE,2),1,j); h = histogram(dCE(:,j),1:nlvls(j));
    set(gca,'xlim',[amin amax]+[-.5 .5]);
    h.BinWidth = .2;
    xlabel([attrKey{j} ' level']); set(gca,'xtick',1:nlvls(j));
end

