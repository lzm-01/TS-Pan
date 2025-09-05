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
     dim_cut = 1;% Cut Final Image
     thvalues = 0;% Threshold values out of dynamic range
     printEPS = 0;% Print Eps
     ratio = 4;% Resize Factor
     L = 11;% Radiometric Resolution

%% ==========Read Data and sensors' info====================
%% read the test dataset; if use your test dataset, please update in this folder
%file_test = '1_TestData/WV3/full/Test(HxWxC)_wv3_data_fr2.mat';

% get I_MS_LR, I_MS, I_PAN and sensors' info; 
%load(file_test)  
i = 1;
NumIndexes = 3;
MatrixResults = zeros(20,NumIndexes);
while(i<=20)
    %file_test = strcat('1_TestData/WV3/full/Test(HxWxC)_wv3_data_fr',i,'.mat');
    % load(strcat('1_TestData/WV2/full/Test(HxWxC)_wv2_data_fr', num2str(i), '.mat'))
    load(strcat('1_TestData/WV3/full/Test(HxWxC)_wv3_data_fr', num2str(i), '.mat'))
    % load(strcat('1_TestData/GF2/fr/Test(HxWxC)_gf2_data_fr', num2str(i), '.mat'))
    % load(strcat('1_TestData/QB/fr/Test(HxWxC)_qb_data_fr', num2str(i), '.mat'))
    %NumIndexes = 3;
    %MatrixResults = zeros(numel(i-1),NumIndexes);
    %alg = 0;
    flagQNR = 0; %% Flag QNR/HQNR, 1: QNR otherwise HQNR

    % zoom-in interesting two regions of figure; you may change them
    % according to your requirment

    %location1 = [10 50 1 60];
    %location2 = [140 180 5 60];
    %clear print
    % file_u2net = strcat('2_DL_Result/WV2/stage-one/10/fr220/output_oriExm_',num2str(i-1),'.mat');
    % file_u2net = strcat('2_DL_Result/WV3/stage-one/1229/fr/330/output_oriExm_',num2str(i-1),'.mat');
    % file_u2net = strcat('2_DL_Result/GF2/stage-one/5534/fr/230/output_oriExm_',num2str(i-1),'.mat');
    % file_u2net = strcat('2_DL_Result/QB/stage-one/4655/fr/150/output_oriExm_',num2str(i-1),'.mat');
    file_u2net = strcat('2_DL_Result/WV3_A/no_ms_hf/5599/fr/330/output_oriExm_',num2str(i-1),'.mat');
    load(file_u2net);
    I_u2net = double(sr);

    [D_lambda_u2net,D_S_u2net,QNRI_u2net] = indexes_evaluation_FS(I_u2net,ms,pan,L,thvalues,lms,sensor,ratio,flagQNR);
    MatrixResults(i,:) = [D_lambda_u2net,D_S_u2net,QNRI_u2net];
    MatrixImage(:,:,:,i) = I_u2net;
    
    if size(lms,3) == 4
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
disp(' |====D_lambda====|===D_S===|=====QNR=====|')
MatrixResults
disp('#######################################################')
disp('Display the average performance')
disp(' |====D_lambda====|===D_S===|=====QNR=====|')
MatrixFinalAvg = mean(MatrixResults)
disp('#######################################################')
disp('Display the std performance')
disp(' |====D_lambda====|===D_S===|=====QNR=====|')
MatrixFinalStd = std(MatrixResults)

%% %%%%%%%%%%% End %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%