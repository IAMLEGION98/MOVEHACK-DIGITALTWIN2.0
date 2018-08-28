%% 
% Support Vector Machines
%
%  Instructions
%  ------------
% 
%  This file contains code the following important functions for implementing 
%  Support Vector Machines(SVM):
%
%     gaussianKernel.m
%     dataset3Params.m
%     
%% Initialization
clear ; close all; clc

%% =============== Part 1: Loading and Visualizing Data ================
%  We start by first loading and visualizing the dataset. 
%  The following code will load the dataset into the environment and plot
%  the data.
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data1: 
% You will have X, y in your environment
load('blink_rate.mat','data');
X=data.X;
y=data.y;
Xval=data.Xval;
yval=data.yval;
% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ========== Part 2: Training SVM with RBF Kernel ==========

% We try different values of C and sigma here.We choose the most optimal
% values to visualize the boundary.
% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);
% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

