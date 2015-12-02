addpath(genpath('.'));
%% Configuration
rfsize = [9 9 3];
%numBases=1600;
numBases = 32;
alpha = 0.25;  %% CV-chosen value for soft-threshold function.
lambda = 1.0;  %% CV-chosen sparse coding penalty.
mask = false;
stride = rfsize;
ratio_pr = 1; % the ratio of positive patches and random patches
ratio_tt = 0.6; % the ratio of train samples and test samples

%%%%% Dictionary Training %%%%%
%alg='patches'; %% Use randomly sampled patches.  Test accuracy 79.14%
alg='omp1';   %% Use 1-hot VQ (OMP-1).  Test accuracy 79.96%
%alg='sc';     %% Sparse coding

%%%%% Encoding %%%%%%
encoder='thresh'; encParam=alpha; %% Use soft threshold encoder.
%encoder='sc'; encParam=lambda; %% Use sparse coding for encoder.

%%%%% SVM Parameter %%%%%
switch (encoder)
 case 'thresh'
  L = 0.01; % L=0.01 for 1600 features.  Use L=0.03 for 4000-6000 features.
 case 'sc'
  L = 1.0; % May need adjustment for various combinations of training / encoding parameters.
end

%% Load data
load('annotation1.mat');
load('volume1.mat');
if mask
    load('mask1.mat');
    I = reshape(I(:).*I_mask(:),size(I));
end

%% extract random patches
switch (alg)
    case 'omp1'
        numpatch = 40000;
    case 'sc'
        numpatch = 100000;
    case 'patches'
        numpatch = 50000; % still needed for whitening
end
patches = zeros(numpatch, prod(rfsize));
for i=1:numpatch
  if (mod(i,10000) == 0) 
      fprintf('Extracting patch: %d / %d\n', i, numpatch); 
  end
  r = random('unid', size(I,1) - rfsize(1) + 1);
  c = random('unid', size(I,2) - rfsize(2) + 1);
  s = random('unid', size(I,3) - rfsize(3) + 1);
  patch = I(r:r+rfsize(1)-1,c:c+rfsize(2)-1,s:s+rfsize(3)-1);
  patches(i,:) = patch(:)';
end

% normalize for contrast
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

% ZCA whitening (with low-pass)
C = cov(patches);
M = mean(patches);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
patches = bsxfun(@minus, patches, M) * P;

%% run training
switch alg
    case 'omp1'
        dictionary = run_omp1(patches, numBases, 50);
    case 'sc'
        dictionary = run_sc(patches, numBases, 10, lambda);
    case 'patches'
        dictionary = patches(randsample(size(patches,1), numBases), :);
        dictionary = bsxfun(@rdivide, dictionary, sqrt(sum(dictionary.^2,2)) + 1e-20);
end
%save dictionary.mat dictionary
% show results of training
% show_centroids(dictionary * 5, rfsize); drawnow;

%% extract all features
% extract all patches
% extract overlapping sub-patches into rows of 'patches'
%[patches, count, y] = extract_patches(I, A, rfsize, numpatch, 'stride', stride);
[patches_pos, countp, yp] = extract_patches(I, A, rfsize, 0, 'positive', []);
[patches_ran, countr, yr] = extract_patches(I, A, rfsize, countp*ratio_pr, 'random', []);
patches = [patches_pos; patches_ran];
y = [yp; yr];
% normalize for contrast
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
% whiten
patches = bsxfun(@minus, patches, M) * P;

% extract training feature
% indp = find(y==1);
nump = round(ratio_tt*countp);
% indn = find(y==0);
numr = round(ratio_tt*countr);

trainX = patches([1:nump,countp+1:countp+numr],:);
trainY = y([1:nump,countp+1:countp+numr]);
testX = patches([nump+1:countp,countp+numr+1:end],:);
testY = y([nump+1:countp,countp+numr+1:end]);
clear patches y;

% random order of training data
ind = randperm(length(trainY),length(trainY));
trainX = trainX(ind, :);
trainY = trainY(ind);

% trainXC = extract_features(trainX, dictionary, encoder, encParam);
trainXC = extract_features(trainX, dictionary, encoder, encParam);
clear trainX;

% standardize data
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
clear trainXC;
trainXCs = [trainXCs, ones(size(trainXCs,1),1)]; % intercept term

theta_lr = learn_LRegression(trainXCs, trainY);
labels_lr = predict(theta_lr, trainXCs);

% train classifier using SVM
trainY = trainY + 1;
theta = train_svm(trainXCs, trainY, 1/L);
[val,labels] = max(trainXCs*theta, [], 2);

fprintf('Training evaluation metrics: \n');
[accuracy, precision, recall, f1] = evaluation(labels, trainY, 2, 1);
[accuracy, precision, recall, f1] = evaluation(labels_lr, trainY-1, 1, 0);

%%%%% TESTING %%%%%

%% Load CIFAR test data
% compute testing features and standardize
testXC = extract_features(testX, dictionary, encoder, encParam);
clear testX;
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
clear testXC;
testXCs = [testXCs, ones(size(testXCs,1),1)];

labels_lr = predict(theta_lr, testXCs);

% test and print result
testY = testY + 1;
[val,labels] = max(testXCs*theta, [], 2);

fprintf('Testing evaluation metrics: \n');
[accuracy, precision, recall, f1] = evaluation(labels, testY, 2, 1);
[accuracy, precision, recall, f1] = evaluation(labels_lr, testY-1, 1, 0);

