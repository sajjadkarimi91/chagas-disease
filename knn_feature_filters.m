function [ind_feature, best_proxy_score,best_ratio_pick]  = knn_feature_filters(X_train, Y_train, X_test, Y_test, max_features_to_select, labels_test, prc_thr)


    num_features = size(X_train, 2);
    selected_features = [];
    remaining_features = 1:num_features;
    proxy_scores = zeros(1, max_features_to_select);
    ratio_scores = zeros(1, max_features_to_select);
    ratio_pick = zeros(1, max_features_to_select);

% Forward selection loop
for iter = 1:max_features_to_select

    % Pre-allocate arrays for parallel processing
    num_remaining = length(remaining_features);
    proxy_scores_temp = zeros(num_remaining, 1);
    valid_features = false(num_remaining, 1);

    % Try each remaining feature in parallel
    parfor f_idx = 1:num_remaining
        current_feature = remaining_features(f_idx);

        % Create feature set with current selection + candidate feature
        current_feature_set = [selected_features, current_feature];

        if isempty(current_feature_set)
            proxy_scores_temp(f_idx) = -1;
            valid_features(f_idx) = false;
            continue;
        end

        % Train model with current feature set
        try
            % Use KNN for parallel processing (more stable than naive Bayes)
            knn_model = fitcknn(X_train(:, current_feature_set), Y_train, 'NumNeighbors', 50);

            % Get predictions
            [~, test_prob_knn] = predict(knn_model, X_test(:, current_feature_set));
            test_prob_knn = test_prob_knn(:, 2); % Probability of positive class

            % Calculate proxy score (same as original)
            test_prob_knn = test_prob_knn+10^-8*randn(size(test_prob_knn));
            prc_95 = prctile(test_prob_knn, prc_thr);
            Y_test_5 = Y_test(test_prob_knn >= prc_95);

            if ~isempty(Y_test_5)
                current_proxy_score = sum(Y_test_5 == 1) / sum(labels_test == 1);
                current_ratio = sum(Y_test_5 == 1) / sum(labels_test == 1);
                current_ratio_pick = length(Y_test_5) / length(labels_test);
            else
                current_proxy_score = 0;
                current_ratio = 0;
                current_ratio_pick = 0;
            end

            proxy_scores_temp(f_idx) = current_proxy_score;
            ratio_scores_temp(f_idx) = current_ratio;
            ratio_pick_temp(f_idx) = current_ratio_pick;
            valid_features(f_idx) = true;

        catch
            % Skip this feature if training fails
            proxy_scores_temp(f_idx) = -1;
            valid_features(f_idx) = false;
        end
    end

    % Find the best feature from parallel results
    valid_indices = find(valid_features);
    if ~isempty(valid_indices)
        [best_proxy_score, best_idx] = max(proxy_scores_temp(valid_indices));
        best_feature = remaining_features(valid_indices(best_idx));
        best_ratio = ratio_scores_temp(valid_indices(best_idx));
        best_ratio_pick = ratio_pick_temp(valid_indices(best_idx));
    else
        best_feature = -1;
        best_proxy_score = -1;
    end

    % If no improvement or no valid feature found, stop
    if best_feature == -1 || best_proxy_score <= 0
        break;
    end

    % Add best feature to selected set
    selected_features = [selected_features, best_feature];
    remaining_features(remaining_features == best_feature) = [];

    fprintf('Iteration %d: Selected feature %d, Proxy score: %.4f, Ratio pick: %.4f\n', iter, best_feature, best_proxy_score, best_ratio_pick);

    % Early stopping if no improvement for 5 consecutive iterations
    if iter > 5 && best_proxy_score <= proxy_scores(max(1, iter-5))
        fprintf('Early stopping: No improvement for 5 iterations\n');
        selected_features(end)=[];
        break;
    end

    proxy_scores(iter) = best_proxy_score;
    ratio_scores(iter) = best_ratio;
    ratio_pick(iter) = best_ratio_pick;
end

% Use selected features for final model training
ind_feature = selected_features;

