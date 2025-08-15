function [binary_output,probability_output]=team_run_model(data_record, classification_model, verbose)

classification_model=classification_model.classification_model;

mn_features = classification_model(1,1,1).mn_features;
std_features = classification_model(1,1,1).std_features;

% header=fileread(data_record);
features = mn_features;
lead_names_target = {'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'};
tic

try
    current_features = get_features(data_record,lead_names_target);
    features=current_features;
end

features(isnan(features))=mn_features(isnan(features));

% Normalize features
features = (features - mn_features) ./ std_features;
features(features>5)=5;
features(features<-5)=-5;

features_all = features;

K = size(classification_model,1);
K=1;
C = size(classification_model,2);
B = size(classification_model,3);
threshold_fold = 0;

for fold = 1:K

    ch_features =size(features_all,2)/C;

    for ch = 1:C

        features_test = features_all(:,(ch-1)*ch_features+1:ch*ch_features);

        % Number of bootstrap samples
        for b = 1:B

            svm_model = classification_model(fold,ch,b).svm_model;
            nn_model = classification_model(fold,ch,b).nn_model;
            ens_mdl_tree = classification_model(fold,ch,b).ens_mdl_tree;
            ind_features = classification_model(fold,ch,b).ind_features;
            % threshold_opt = classification_model(fold,ch,b).threshold;
            std_svm = classification_model(fold,ch,b).std_svm;
            std_tree = classification_model(fold,ch,b).std_tree;


            X_test = features_test(:,ind_features);

            est_trgt_nn = predict(nn_model,X_test);
            [~,est_trgt_svm] = predict(svm_model,X_test);
            [~,est_trgt_tree] = predict(ens_mdl_tree,X_test);

            est_trgt_tree = exp(est_trgt_tree(:,2)/(3*std_tree)) ./ ( exp(est_trgt_tree(:,1)/(3*std_tree)) + exp(est_trgt_tree(:,2)/(3*std_tree)) );
            est_trgt_svm = exp(est_trgt_svm(:,2)/(3*std_svm)) ./ ( exp(est_trgt_svm(:,1)/(3*std_svm)) + exp(est_trgt_svm(:,2)/(3*std_svm)) );
            est_trgt_nn = est_trgt_nn(:,2);


            fused_prob = (est_trgt_nn+est_trgt_svm+est_trgt_tree)/3;


            if ch==1 && b==1 && fold==1
                test_prob = fused_prob;
            else
                test_prob = test_prob + fused_prob;
            end

        end


    end

    threshold_fold = threshold_fold + classification_model(fold,1,1).threshold_fold/K;
end

% Average the probabilities
test_prob = test_prob/(C*B*K);

predicted_class = test_prob>threshold_fold;

toc

if str2double(predicted_class)==0
    binary_output='False';
else
    binary_output='True';
end

probability_output=test_prob;

end