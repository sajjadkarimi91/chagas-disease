function [nn_model,compact_svm,compact_tree,tree_params] = train_ml_models(X_train,Y_train,X_test,Y_test,X_test_nn,Y_test_nn,tree_params)

if nargin<7
    tree_params = [];
end

% Train Neural Network model
numResponses = 2;
numChannels = size(X_train,2);
layers = [ featureInputLayer(numChannels, Normalization="zscore")
    fullyConnectedLayer(100)
    clippedReluLayer(1)
    dropoutLayer(0.1)
    fullyConnectedLayer(25)
    leakyReluLayer(0.001)
    fullyConnectedLayer(numResponses)
    softmaxLayer()];

try
    metric_nn = aucMetric(AverageType="macro");
    options = trainingOptions("adam", ...
        MaxEpochs=5, ...
        MiniBatchSize = 512,...
        ValidationData={X_test_nn, categorical(Y_test_nn)},...
        OutputNetwork="best-validation-loss", ...
        Metrics= metric_nn,...
        InitialLearnRate=0.001, ...
        L2Regularization = 0.00001,...
        Shuffle = "every-epoch", ...
        Verbose= false);

    nn_model = trainnet(X_train, categorical(Y_train), layers, 'crossentropy',options);
catch

    metric_nn = aucMetric(AverageType="macro");
    options = trainingOptions("adam", ...
        MaxEpochs=10, ...
        MiniBatchSize = 1024,...
        ValidationData={X_test_nn, categorical(Y_test_nn)},...
        OutputNetwork="best-validation-loss", ...
        Metrics= metric_nn,...
        InitialLearnRate=0.001, ...
        L2Regularization = 0.00001,...
        Shuffle = "every-epoch", ...
        Verbose= false);

    nn_model = trainnet(X_train, categorical(Y_train), layers, 'crossentropy',options);

end
svm_model = fitcsvm(X_train, categorical(Y_train),'KernelFunction','linear','CacheSize','maximal','BoxConstraint',1,'KernelScale','auto');
compact_svm = compact(svm_model);
compact_svm = discardSupportVectors(compact_svm);
% Train Ensemble models

if isempty(tree_params)
    learnRate = [0.1 0.2];
    numLR = numel(learnRate);
    maxNumSplits = 2.^(3:4);
    numMNS = numel(maxNumSplits);
    numTrees = [20 30];
    clear auc_tree_all

    for i = 1:numel(numTrees)
        for k = 1:numLR
            for j = 1:numMNS
                % disp([k,j])
                t = templateTree('MaxNumSplits',maxNumSplits(j),'Prune','on');
                Mdl = fitcensemble(X_train,Y_train,'Method','AdaBoostM1','NumLearningCycles',numTrees(i),...
                    'Learners',t,'LearnRate',learnRate(k));
                [~,est_trgt_tree] = predict(Mdl,X_test);
                [~,~,~,auc_tree] = perfcurve(Y_test,est_trgt_tree(:,2),1);
                auc_tree_all(i,j,k) = auc_tree;
            end
        end
    end


    [~,ind_max] = max(auc_tree_all(:));
    [idxNumTrees,idxMNS,idxLR] = ind2sub(size(auc_tree_all),ind_max);

    tree_params.maxNumSplits = maxNumSplits(idxMNS);
    tree_params.numTrees = numTrees(idxNumTrees);
    tree_params.learnRate = learnRate(idxLR);

end


t = templateTree('MaxNumSplits',tree_params.maxNumSplits,'Prune','on');
ens_mdl_tree = fitcensemble(X_train,Y_train,'Method','AdaBoostM1','NumLearningCycles',tree_params.numTrees,...
    'Learners',t,'LearnRate',tree_params.learnRate);

compact_tree = compact(ens_mdl_tree);