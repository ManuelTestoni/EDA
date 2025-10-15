%% import excel by command lines
[olive, text]=xlsread('olive_oil'); % olive_oil.xlsx è il nome del file da cambiare quando si cambia file
obj_lab=text(2:383,1);
var_lab=text(1,2:end);
for i=1:7; olive_lab{i}=deblank(var_lab{i}); end

