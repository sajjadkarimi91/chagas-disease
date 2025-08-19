function team_train_model(input_directory,output_directory, verbose)

if verbose>=1
    disp('Finding Challenge data...')
end

aws_run = 1;

if aws_run>0
    % Find the recordings
    records=dir(fullfile(input_directory,'**','*.hea'));
    % records = records(1:100:end);
    num_records = length(records);

    if num_records<1
        error('No records were provided')
    end

    if verbose>=1
        disp('Extracting features and labels from the data...')
    end

    if ~isdir(output_directory)
        mkdir(output_directory)
    end


    original_path=pwd;
    if ~isdeployed addpath(genpath(original_path)); end

    lead_names_target = {'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'};
    rng(42)
    warning('off')

    parfor j=1:50
        dummy_ab(j)=rand(1);
    end

    tic
    parfor j=1:50

        if verbose>1
            fprintf('%d/%d \n',j,num_records)
        end

        file_load = fullfile(records(j).folder,records(j).name);

        header=fileread(fullfile(records(j).folder,records(j).name));
        %
        % if ~strcmp(pwd,records(j).folder)
        %     cd(records(j).folder)
        % end
        % Get labels
        labels(j)=get_labels(header);
        if labels(j)==0 && rand<0.7
            features_temp(j,:)=nan(1,118*length(lead_names_target));
        else
            % Extract features
            try
                current_features = get_features(file_load,lead_names_target);
                features_temp(j,:)=current_features;
            catch
                features_temp(j,:)=nan;
            end
        end


    end

    ratio_selection = 0.35;
    time_50 = toc;
    total_records = (8000 + ratio_selection * (num_records-8000))/50 * time_50;
    fprintf('Estimated time to complete: %f hours \n', total_records/3600)

    parfor j=1:num_records

        if verbose>1
            fprintf('%d/%d \n',j,num_records)
        end

        file_load = fullfile(records(j).folder,records(j).name);

        header=fileread(fullfile(records(j).folder,records(j).name));
        %
        % if ~strcmp(pwd,records(j).folder)
        %     cd(records(j).folder)
        % end
        % Get labels
        labels(j)=get_labels(header);
        if labels(j)==0 && rand<(1-ratio_selection)
            features(j,:)=nan(1,118*length(lead_names_target));
        else
            % Extract features
            try
                current_features = get_features(file_load,lead_names_target);
                features(j,:)=current_features;
            catch
                features(j,:)=nan;
            end
        end


    end

    cd(original_path)
    % rmpath(genpath(original_path))

    % save('features_all.mat')

end


%%

% load('features_all.mat')
ind_fullnan = all(isnan(features),2);
features = features(~ind_fullnan,:);
labels = labels(~ind_fullnan);

labels = labels(:);
classes=sort(unique(labels));


fprintf('Training the model on the data... \n')


mn_features = median(features,1,'omitnan');
for i=1:size(features,2)
    isnan_idx=isnan(features(:,i));
    features(isnan_idx,i)=mn_features(i);
end

std_features = std(features,1,'omitnan');

% Normalize features
features = (features - mn_features) ./ std_features;
features(features>5)=5;
features(features<-5)=-5;

rng(42)
% Number of folds for cross-validation
K = 5;
% cv_splits = cvpartition(length(labels), 'Kfold',k);
cv_splits = cvpartition(labels,'Kfold',K,'stratify',true);

features_all = features;

%%

fold = 1;
% Get training and testing indices for this fold
train_idx = training(cv_splits,fold);
test_idx = test(cv_splits,fold);

labels_train = labels(train_idx);
labels_test = labels(test_idx);

features = features_all;
features_train = features(train_idx,:);
features_test = features(test_idx,:);

features_train_1 = features_train(labels_train==1,:);
features_train_0 = features_train(labels_train==0,:);
features_train_0_b = features_train_0;

clear stat_ks ind_pass
for f = 1:size(features,2)

    x1 = features_train_1(:,f);
    x2 = features_train_0_b(:,f);
    x_all = [x1;x2];
    y_all = [ones(size(x1,1),1); zeros(size(x2,1),1)];
    x_all = x_all+10^-8*randn(size(x_all));

    prc_95(f,1) = prctile(x2, 95);
    y_all_5 = y_all(x_all >= prc_95(f,1));
    s_95 = sum(y_all_5 == 1) / sum(y_all == 1);

    prc_5(f,1) = prctile(x2, 5);
    y_all_5 = y_all(x_all <= prc_5(f,1));
    s_5 = sum(y_all_5 == 1) / sum(y_all == 1);
    if s_95>s_5
        stat_ks(f,1) = s_95;
        sign_score(f) = 1;
        prc_score(f) = prc_95(f,1);
        ind_pass{f,1} = find(x_all >= prc_95(f,1));

    else
        stat_ks(f,1) = s_5;
        sign_score(f) = -1;
        prc_score(f) = prc_5(f,1);
        ind_pass{f,1} = find(x_all <= prc_5(f,1));
    end



end

% [~,ind_features] = sort(stat_ks,'descend');
ind_features = find(stat_ks>0.12);

ind_pass_unique = [];
for f= 1:size(ind_features,1)
    ind_pass_unique = unique([ind_pass_unique; ind_pass{ind_features(f),1}]);
end
% ind_features_ks = ind_features(1:200);

features_train_1 = features_train_1(:,ind_features);
features_train_0_b = features_train_0_b(:,ind_features);
% features_test = features_test(:,ind_features);

X_train = [features_train_1; features_train_0_b];
Y_train = [ones(size(features_train_1,1),1); zeros(size(features_train_0_b,1),1)];
X_train = X_train(ind_pass_unique,:);
Y_train = Y_train(ind_pass_unique);

sign_score_selected = sign_score(ind_features);
prc_score_selected = prc_score(ind_features);

x_all = sum(X_train.*sign_score_selected(:)',2);
prc_95 = prctile(x_all, 95);
y_all_5 = Y_train(x_all >= prc_95);
stat_ks_avg = sum(y_all_5 == 1) / sum(y_all == 1)

clear  ind_pass
ind_pass_unique_test = [];

for f = 1:length(ind_features)

    x_all = features_test(:,ind_features(f));

    if  sign_score(ind_features(f))>0
        ind_pass{f,1} = find(x_all >= prc_score(ind_features(f)));
    else
        ind_pass{f,1} = find(x_all <= prc_score(ind_features(f)));
    end

    ind_pass_unique_test = unique([ind_pass_unique_test; ind_pass{f,1}]);

end

features_test_passed = features_test(ind_pass_unique_test,:);
features_test_passed = features_test_passed(:,ind_features);
final_prob = zeros(length(labels_test),1);
labels_test_passed = labels_test(ind_pass_unique_test);

%%


rng(42)

K=1;
for fold = 1:1
    % Get training and testing indices for this fold

    labels_train = Y_train;
    features_train = X_train;

    ch = 1;

    features_train_1 = features_train(labels_train==1,:);
    features_train_0 = features_train(labels_train==0,:);

    minority_sample = 3*size(features_train_1,1);
    rand_idx = randperm(sum(labels_train==0));
    B = floor(sum(labels_train==0)/minority_sample);
    % Number of bootstrap samples
    B = min(B,5);
    for b = 1:B
        tic
        disp([fold,ch,b])

        features_train_0_b = features_train_0(rand_idx((b-1)*minority_sample+1:b*minority_sample),:);
        clear stat_ks
        for f = 1:size(features_train,2)

            x1 = features_train_1(:,f);
            x2 = features_train_0_b(:,f);
            [h,pval_ks,stat_ks(f)] = kstest2(x1,x2);

        end

        ind_features_ks = find(stat_ks>median(stat_ks,'omitmissing')-2);

        X_train = [features_train_1(:,ind_features_ks); features_train_0_b(:,ind_features_ks)];
        Y_train = [ones(size(features_train_1,1),1); zeros(size(features_train_0_b,1),1)];


        % Shuffle training data before training to avoid single-class mini-batches
        numSamples = size(X_train, 1);
        idx = randperm(numSamples);
        X_train = X_train(idx, :);
        Y_train = Y_train(idx, :);

        label_test_chagas = labels_test_passed==1;
        X_test_nn = [features_test_passed(:,ind_features_ks); repmat(features_test_passed(label_test_chagas,:),floor(B/2),1)];
        Y_test_nn = [labels_test_passed; repmat(labels_test_passed(label_test_chagas),floor(B/2),1)]; % 0 for negative samples

        X_test = features_test_passed(:,ind_features_ks);
        Y_test = labels_test_passed; % 0 for negative samples

        X_test_1 = X_test(labels_test_passed==1,:);
        Y_test_1 = Y_test(labels_test_passed==1);

        numSamples = size(X_test_nn, 1);
        idx = randperm(numSamples);
        X_test_nn = X_test_nn(idx, :);
        Y_test_nn = Y_test_nn(idx, :);

        weihgts_train = ones(length(Y_train),1);
        weihgts_train(Y_train==1)= sum(Y_train==0) /sum(Y_train==1);

        [nn_model,compact_svm,compact_tree,tree_params] = train_ml_models(X_train,Y_train,X_test,Y_test,X_test_nn,Y_test_nn);

        est_trgt_nn = predict(nn_model,X_test);
        [~,est_trgt_svm] = predict(compact_svm,X_test);
        std_svm = std(est_trgt_svm(:,2));

        [~,est_trgt_tree] = predict(compact_tree,X_test);
        std_tree = std(est_trgt_tree(:,2));

        est_trgt_tree = exp(est_trgt_tree(:,2)/(3*std_tree)) ./ ( exp(est_trgt_tree(:,1)/(3*std_tree)) + exp(est_trgt_tree(:,2)/(3*std_tree)) );
        est_trgt_svm = exp(est_trgt_svm(:,2)/(3*std_svm)) ./ ( exp(est_trgt_svm(:,1)/(3*std_svm)) + exp(est_trgt_svm(:,2)/(3*std_svm)) );
        est_trgt_nn = est_trgt_nn(:,2);

        [spec_x,sen_y,T,auc_nn] = perfcurve(Y_test,est_trgt_nn,1);
        [spec_x,sen_y,T,auc_svm] = perfcurve(Y_test,est_trgt_svm,1);
        [spec_x,sen_y,T,auc_tree] = perfcurve(Y_test,est_trgt_tree,1);

        fused_prob = (est_trgt_nn+est_trgt_svm+est_trgt_tree)/3;

        [spec_x,sen_y,T,auc_this] = perfcurve(Y_test,fused_prob,1);
        [~,ind_max] = max((1-spec_x).*sen_y);
        threshold_opt = T(ind_max);

        if aws_run>0
            [nn_model,compact_svm,compact_tree] = train_ml_models([X_train;X_test_1],[Y_train;Y_test_1],X_test,Y_test,X_test_nn,Y_test_nn,tree_params);
        end
        classification_model(fold,ch,b).svm_model = compact_svm;
        classification_model(fold,ch,b).nn_model = nn_model;
        classification_model(fold,ch,b).ens_mdl_tree = compact_tree;
        classification_model(fold,ch,b).ind_features = ind_features;
        classification_model(fold,ch,b).mn_features = mn_features;
        classification_model(fold,ch,b).threshold = threshold_opt;
        classification_model(fold,ch,b).std_features = std_features;
        classification_model(fold,ch,b).auc = auc_this;
        classification_model(fold,ch,b).auc_nn = auc_nn;
        classification_model(fold,ch,b).auc_svm = auc_svm;
        classification_model(fold,ch,b).auc_tree = auc_tree;
        classification_model(fold,ch,b).std_svm = std_svm;
        classification_model(fold,ch,b).std_tree = std_tree;
        classification_model(fold,ch,b).prc_score = prc_score;
        classification_model(fold,ch,b).sign_score = sign_score;
        classification_model(fold,ch,b).ind_features_ks = ind_features_ks;


        if ch==1 && b==1
            test_prob = fused_prob;
        else
            test_prob = test_prob + fused_prob;
        end

        toc


    end

    Y_test = labels_test;
    final_prob(ind_pass_unique_test) = test_prob;
    % Average the probabilities
    test_prob_auc = final_prob;
    [spec_x,sen_y,T,auc_all(fold)] = perfcurve(Y_test,test_prob_auc,1);
    [~,ind_max] = max((1-spec_x).*sen_y);
    threshold_opt = T(ind_max);

    classification_model(fold,1,1).auc_fold = auc_all(fold);
    classification_model(fold,1,1).threshold_fold = threshold_opt;

    prc_95 = prctile(test_prob_auc,95);
    Y_test_5 = Y_test(test_prob_auc>prc_95);
    disp(auc_all)
    proxy_score(fold) = sum(Y_test_5==1)/sum(Y_test==1);
    disp(proxy_score)
    classification_model(fold,1,1).proxy_score = proxy_score(fold);

end



save_models(output_directory, classification_model, classes)

if verbose>=1
    fprintf('Done. \n')
end

end

function save_models(output_directory, classification_model, classes)

filename = fullfile(output_directory,'classification_model.mat');
save(filename,'classification_model','classes','-v7.3');
end

function label=get_labels(header)

header=strsplit(header,'\n');
dx=header(startsWith(header,'# Chagas'));
if ~isempty(dx)
    dx=strsplit(dx{1},':');
    dx=strtrim(dx{2});

    if startsWith(dx,'Fa')
        label=0;
    else
        label=1;
    end
else
    error('# Labels missing!')
end

end