%%%%%%%%%%%%%%%%%%%%%%%%%%%For FUll-Resolution%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  1) This is a test demo to show all full-resolution results of traditional and DL methods
%     Here, we take WV3 test dataset as example. Readers can change the corresponding director 
%     and setting to test other/your datasets
%  2) The codes of traditional methods are from the "pansharpening toolbox for distribution",
%     thus please cite the paper:
%     [1] G. Vivone, et al., A new benchmark based on recent advances in multispectral pansharpening: Revisiting
%         pansharpening with classical and emerging pansharpening methods, IEEE Geosci. Remote Sens. Mag., 
%         9(1): 53–81, 2021
%  3) Also, if you use this toolbox, please cite our paper:
%     [2] L.-J. Deng, et al., Machine Learning in Pansharpening: A Benchmark, from Shallow to Deep Networks, 
%         IEEE Geosci. Remote Sens. Mag., 2022

%  LJ Deng (UESTC), 2020-02-27

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: the test dataset of full-resolution are too huge to upload to
% GitHub, thus we provide cloud links to readers to download them to
% successfully run this demo, including:

% i) Download link for full-resolution WV3-NewYork example (named "NY1_WV3_FR.mat"):
%     http:********   (put into the folder of "1_TestData/Datasets Testing")

% ii) Download link of DL's results for full-resolution WV3-NewYork example:
%     http:********   (put into the folder of "'2_DL_Result/WV3")

% Once you have above datasets, you can run this demo successfully, then
% understand how this demo run!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;
%% =======load directors========
% Tools
addpath([pwd,'/Tools']);

% Select algorithms to run
algorithms = {'U2Net'};

% director to save EPS figures for latex editing; if other dataset, please
% change the director correspondingly
data_name = '3_EPS/WV3/wv3_os_fr';  
% (Note: If there is no sensor's info in your dataset, 
% please find and update these info in the following commented lines):

%------ following are sensor's info for WV3 (an example for WV3)----
     sensor = 'WV3';
     Qblocks_size = 32;
     bicubic = 0;% Interpolator
     flag_cut_bounds = 1;% Cut Final Image
     dim_cut = 21;% Cut Final Image
     thvalues = 0;% Threshold values out of dynamic range
     printEPS = 0;% Print Eps
     ratio = 4;% Resize Factor
     L = 11;% Radiometric Resolution

%% ==========Read Data and sensors' info====================
%% read the test dataset; if use your test dataset, please update in this folder
%file_test = '1_TestData/WV3/full/Test(HxWxC)_wv3_data_fr2.mat';

% get I_MS_LR, I_MS, I_PAN and sensors' info; 
%load(file_test) 
const = 2047;
% const = 1023;
i = 1;
NumIndexes = 6;
MatrixResults = zeros(20,NumIndexes);
while(i<=20)
    %file_test = strcat('1_TestData/WV3/full/Test(HxWxC)_wv3_data_fr',i,'.mat');
    load(strcat('1_TestData/WV3/rr/Test(HxWxC)_wv3_data', num2str(i), '.mat'))
    % load(strcat('1_TestData/WV2/rr/Test(HxWxC)_wv2_data', num2str(i), '.mat'))
    % load(strcat('1_TestData/GF2/rr/Test(HxWxC)_gf2_data', num2str(i), '.mat'))
    % load(strcat('1_TestData/QB/rr/Test(HxWxC)_qb_data', num2str(i), '.mat'))
    
    %NumIndexes = 3;
    %MatrixResults = zeros(numel(i-1),NumIndexes);
    %alg = 0;
    flagQNR = 0; %% Flag QNR/HQNR, 1: QNR otherwise HQNR

    % zoom-in interesting two regions of figure; you may change them
    % according to your requirment

    %location1 = [10 50 1 60];
    %location2 = [140 180 5 60];
    %clear print
    % file_u2net = strcat('2_DL_Result/WV2/dense/2956/rr/330/output_mulExm_',num2str(i-1),'.mat');
    % file_u2net = strcat('2_DL_Result/WV3/stage-one/1229/rr/330/output_mulExm_',num2str(i-1),'.mat');
    file_u2net = strcat('2_DL_Result/WV3_A/no_ms_hf/5599/rr/330/output_mulExm_',num2str(i-1),'.mat');
    % file_u2net = strcat('2_DL_Result/GF2/stage-one/5022/rr/240/output_mulExm_',num2str(i-1),'.mat');
    % file_u2net = strcat('2_DL_Result/GF2_QB/adwm/2964/rr/500/output_mulExm_',num2str(i-1),'.mat');
    % file_u2net = strcat('2_DL_Result/QB/stage-one/4655/rr/150/output_mulExm_',num2str(i-1),'.mat');
    % file_u2net = strcat('2_DL_Result/QB_GF2/stage-one/5022/rr/140/output_mulExm_',num2str(i-1),'.mat');
    load(file_u2net);
    I_net = double(sr);

    [Q_avg_net, SAM_net, ERGAS_net, SCC_net, Q_net] = indexes_evaluation(I_net,gt,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    [PSNR_net] = quality_assess(sr / const, gt / const);
    MatrixResults(i,:) = [Q_net,Q_avg_net,SAM_net,ERGAS_net,SCC_net, PSNR_net];
    MatrixImage(:,:,:,i) = I_net;
    
    
    if size(gt,3) == 4
        vect_index_RGB = [3,2,1];
    else
        vect_index_RGB = [5,3,2];
    end


    i = i + 1;
end
%% %%%%%%%%%%% Show and Save Results %%%%%%%%%%%%%%%%%%%%%%%%%

titleImages = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
% figure, showImagesAll(MatrixImage,titleImages,vect_index_RGB,flag_cut_bounds,dim_cut,0);

%% ======Display the final performance =======
fprintf('\n')
disp('#######################################################')
disp(['Display the performance for:'])
disp('#######################################################')
disp(' |====Q====|===Q_avg===|=====SAM=====|======ERGAS=======|=======SCC=======|======PSNR=======|')
MatrixResults
disp('#######################################################')
disp('Display the average performance')
disp(' |====Q====|===Q_avg===|=====SAM=====|======ERGAS=======|=======SCC=======|======PSNR=======|')
MatrixFinalAvg = mean(MatrixResults)
disp('#######################################################')
disp('Display the std performance')
disp(' |====Q====|===Q_avg===|=====SAM=====|======ERGAS=======|=======SCC=======|======PSNR=======|')
MatrixFinalStd = std(MatrixResults)

%% %%%%%%%%%%% End %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%