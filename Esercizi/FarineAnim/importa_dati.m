% Import wines.xls and prepare a dataset for PLS_Toolbox
evrimovepath('top'); % to put PLS toolbox first on the path
dati = importdata('wines_condescrvar.xls'); % import the file 
datix = dati.data.WINES; % extract the numeric part
txt = dati.textdata.WINES; % extract the text part
wines_ds = dataset(datix); % create a dataset

% Add class index
wines_ds.classid{1,1} = cell2str(txt(2:end,1)); 
% Add variable names
wines_ds.label{2} = cell2str(txt(1,2:14)');

% Save the dataset using a valid filename
save('wines_ds.mat', 'wines_ds');
