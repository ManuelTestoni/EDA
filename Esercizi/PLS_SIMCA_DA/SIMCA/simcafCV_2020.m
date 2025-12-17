function model=simcafCV(varargin)
%---------------------------------------------------------
% SIMCA function.
% Computes a SIMCA model for the classification of samples
% with the possibility of cross-validation.
%
% Requires a dataset object as input (X) and the maximum number of 
% components (ncomp) to be tested for SIMCA modeling.
% The number of components (ncomp) can be either a single 
% number, so that for all classes are tested  the same 
% dimensionality, or a vector having the same dimension as the
% number of classes.
%
% Default options are obtained by digiting opt=simcaf('options')
%
%% preprocessing options:
% options.preproc= ......
% 'auto'  autoscaling
% 'mean' mean centering
% 'none'  no preprocessing
% 'autop' pareto scaling + mean center
% 'scale' scaling for standard deviation (no centering)
% 'scalep' pareto scaling (no centering)
%% confidence lim options
% options.lim=....
% 'jm' jackson mudholkar and hotelling-tsquare
% 'prct' percentile of T2 and Q
%% CV options
% options.cv.cvtype=....
% 'full'   leave one out
%'syst111'  contiguous groups
%'syst123'  venetian blind
%'random'  random groups
% 'manual'  custom choice of segments
% 'none'  no CV applied
% options.cv.cvsegments = .....
% for 'full' and 'none' leave empty
% for 'random' 'syst111' 'syst123'  give the number of segments (splits)
% fro 'manual' enter the name of a cell array with as many cells as number
% of segments (splits) in each cell a vector with the number of the samples
% to exclude in that segment
% 'none'  
% I/O: model=simcaf(X,ncomp,options)

clc
close all


if nargin==3
    X=varargin{1};
    ncomp=varargin{2};
    options=varargin{3};
    model=simcaf_mainCV(X,ncomp,options);
else
    if nargin==2
        X=varargin{1};
        ncomp=varargin{2};
        options=struct('cv',[],'preproc','auto','plots','off','lim','jm','crit','combined','cl',0.95);
        options.cv.cvtype='none';
        options.cv.cvsegments=[];
        model=simcaf_main(X,ncomp,options);
    elseif nargin==1 && strcmp(varargin{1},'options')
            model=struct('cv',[],'preproc','auto','plots','off','lim','jm','crit','combined','cl',0.95);
            model.cv.cvtype='none';
            model.cv.cvsegments=[];
    end
end
