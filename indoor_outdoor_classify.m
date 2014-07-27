clc;
clear;
%%
addpath(genpath(fullfile('.','')));
%vl_setup;
%%
img_dir = 'img/ColorCheckerDatabase/srgb8bit/';
hist_dir = 'feat/';
voc_dir = 'vocabulary/';
result_dir = 'results/';
seg_dir = 'seg/';
img_file_names = dir(fullfile(img_dir, '*.tif'));
num_file_names = size(img_file_names, 1);
img_names = cell(num_file_names,1);
outdoor_rates = [];
indoor_rates = [];
for i = 1 : num_file_names
    img_file_name = img_file_names(i).name;
    [img_file_dir img_name img_suffix] = fileparts(img_file_name);
    img_names{i} = img_name;
end
num_words = 100;
voc_path = [voc_dir filesep 'segment_voc' num2str(num_words) '.mat'];
if ~exist(voc_path, 'file')
    F = getFeatures_multiseg(hist_dir, seg_dir, img_names); %
    textonfeats = F.textonfeats;
    colorfeats = F.colorfeats;
    phogfeats = F.phogfeats;
    textonfeats = normalizeFeats(textonfeats);                          %
    colorfeats = normalizeFeats(colorfeats);
    phogfeats = normalizeFeats(phogfeats);
    feats = zeros(size(textonfeats,1), size(textonfeats,2) + size(colorfeats,2) + size(phogfeats,2));
    feats(:,1:size(textonfeats,2)) = textonfeats;
    feats(:,size(textonfeats,2) + 1:size(textonfeats,2) + size(colorfeats,2)) = colorfeats;
    feats(:,size(textonfeats,2) + size(colorfeats,2) + 1 : end) = phogfeats;
    feats = vl_colsubset(feats', 10e3);                                        %
    feats = single(feats);
    [clusters, assigns] = vl_kmeans(feats, num_words, 'verbose', 'algorithm', 'elkan');%
    kdtrees = vl_kdtreebuild(clusters);%
    save(voc_path, 'clusters','kdtrees', 'num_words');
else
    load(voc_path);
end

cvfile = 'threefoldCVsplit.mat';
if ~exist(cvfile,'file')
    outdoor = load_outdoor_label(img_names);
    Ltrue = load_illuminants(img_dir,img_names);

    ind = find(outdoor==1);
    ind = ind(randperm(numel(ind)));
    te_split{1} = ind(1:107);
    te_split{2} = ind(108:215);
    te_split{3} = ind(216:end);
    ind = find(outdoor==0);
    ind = ind(randperm(numel(ind)));
    te_split{1} = [te_split{1},ind(1:82)];
    te_split{2} = [te_split{2},ind(83:165)];
    te_split{3} = [te_split{3},ind(166:end)];
    
    ind = 1:numel(outdoor);
    for i=1:3
        tr_split{i} = setdiff(ind,te_split{i});
    end
    save(cvfile,'tr_split','te_split','img_names')
else
    load(cvfile,'te_split','tr_split','img_names');
    outdoor = load_outdoor_label(img_names);
    Ltrue = load_illuminants(img_dir,img_names);
end
pred_outdoor = zeros(num_file_names,1);
for i=1:3
    tr_files = img_names(tr_split{i});
    F = getFeatures_multiseg(hist_dir, seg_dir, tr_files);
    textonfeats = F.textonfeats;
    colorfeats = F.colorfeats;
    phogfeats = F.phogfeats;
    imlabels = F.imlabels;
    textonfeats = normalizeFeats(textonfeats);
    colorfeats = normalizeFeats(colorfeats);
    phogfeats = normalizeFeats(phogfeats);
    feats = zeros(size(textonfeats,1), size(textonfeats,2) + size(colorfeats,2) + size(phogfeats,2));
    feats(:,1:size(textonfeats,2)) = textonfeats;
    feats(:,size(textonfeats,2) + 1:size(textonfeats,2) + size(colorfeats,2)) = colorfeats;
    feats(:,size(textonfeats,2) + size(colorfeats,2) + 1 : end) = phogfeats;
    outdoor_label = outdoor(tr_split{i});
    tr_out_files = tr_files(outdoor_label == 1);
    X_fg = zeros(num_words,size(tr_out_files,1));
    for j = 1 : size(tr_out_files,1)
        img_name = tr_out_files{j};
        imfeats = [];
        for m = 1 : length(imlabels)
            if strcmp(img_name, imlabels{m})
                imfeats = [imfeats;feats(m,:)];
            end
        end
        imfeats = single(imfeats);
        binsa = double(vl_kdtreequery(kdtrees, clusters, imfeats',  'MaxComparisons', 15));
        hist = zeros(num_words, 1);
        hist = vl_binsum(hist, ones(size(binsa)), binsa);
        hist = single(hist/sum(hist));
        X_fg(:,j) = hist';   
    end
    tr_in_files = tr_files(outdoor_label == 0);
    X_bg = zeros(num_words,size(tr_in_files,1));
    for j = 1 : size(tr_in_files,1)
        img_name = tr_in_files{j};
        imfeats = [];
        for m = 1 : length(imlabels)
            if strcmp(img_name, imlabels{m})
                imfeats = [imfeats;feats(m,:)];
            end
        end
        imfeats = single(imfeats);
        binsa = double(vl_kdtreequery(kdtrees, clusters, imfeats',  'MaxComparisons', 15));
        hist = zeros(num_words, 1);
        hist = vl_binsum(hist, ones(size(binsa)), binsa);
        hist = single(hist/sum(hist));
        X_bg(:,j) = hist';   
    end
    
    %% positive 
    Pw_pos = (1 + sum(X_fg,2)) / (num_words + sum(sum(X_fg)));

    %% positive 
    Pw_neg = (1 + sum(X_bg,2)) / (num_words + sum(sum(X_bg)));
    %%% Compute posterior probability of each class given likelihood models
    %%% assume equal priors on each class
    class_priors = [0.5 0.5];

    %% positive is index 1
    %% negitive class is index 2

    %%%% do everything in log-space for numerical reasons....

    %%% positive model on positive training images
    Pc_d_pos_train = [];
    for a=1:size(tr_out_files,1)
        Pc_d_pos_train(1,a) = log(class_priors(1)) + sum(X_fg(:,a) .* log(Pw_pos)); 
    end

    %%% negative model on positive training images
    for a=1:size(tr_out_files,1)
        Pc_d_pos_train(2,a) = log(class_priors(2)) + sum(X_fg(:,a) .* log(Pw_neg)); 
    end

    %%% positive model on negative training images
    Pc_d_neg_train = [];
    for a=1:size(tr_in_files,1)
        Pc_d_neg_train(1,a) = log(class_priors(1)) + sum(X_bg(:,a) .* log(Pw_pos)); 
    end

    %%% negative model on negitive training images
    for a=1:size(tr_in_files,1)
        Pc_d_neg_train(2,a) = log(class_priors(2)) + sum(X_bg(:,a) .* log(Pw_neg)); 
    end
    te_files = img_names(te_split{i});
    F = getFeatures_multiseg(hist_dir, seg_dir, te_files);
    textonfeats = F.textonfeats;
    colorfeats = F.colorfeats;
    phogfeats = F.phogfeats;
    imlabels = F.imlabels;
    textonfeats = normalizeFeats(textonfeats);
    colorfeats = normalizeFeats(colorfeats);
    phogfeats = normalizeFeats(phogfeats);
    feats = zeros(size(textonfeats,1), size(textonfeats,2) + size(colorfeats,2) + size(phogfeats,2));
    feats(:,1:size(textonfeats,2)) = textonfeats;
    feats(:,size(textonfeats,2) + 1:size(textonfeats,2) + size(colorfeats,2)) = colorfeats;
    feats(:,size(textonfeats,2) + size(colorfeats,2) + 1 : end) = phogfeats;
    
    
    te_outdoor_label = outdoor(te_split{i});
    te_out_files = te_files(te_outdoor_label == 1);
    X_fg = zeros(num_words,size(te_out_files,1));
    for j = 1 : size(te_out_files,1)
        img_name = te_out_files{j};
        imfeats = [];
        for m = 1 : length(imlabels)
            if strcmp(img_name, imlabels{m})
                imfeats = [imfeats;feats(m,:)];
            end
        end
        imfeats = single(imfeats);
        binsa = double(vl_kdtreequery(kdtrees, clusters, imfeats',  'MaxComparisons', 15));
        hist = zeros(num_words, 1);
        hist = vl_binsum(hist, ones(size(binsa)), binsa);
        hist = single(hist/sum(hist));
        X_fg(:,j) = hist';   
    end
   
    te_in_files = te_files(te_outdoor_label == 0);
    X_bg = zeros(num_words,size(te_in_files,1));
    for j = 1 : size(te_in_files,1)
        img_name = te_in_files{j};
        imfeats = [];
        for m = 1 : length(imlabels)
            if strcmp(img_name, imlabels{m})
                imfeats = [imfeats;feats(m,:)];
            end
        end
        imfeats = single(imfeats);
        binsa = double(vl_kdtreequery(kdtrees, clusters, imfeats',  'MaxComparisons', 15));
        hist = zeros(num_words, 1);
        hist = vl_binsum(hist, ones(size(binsa)), binsa);
        hist = single(hist/sum(hist));
        X_bg(:,j) = hist';   
    end
    %%%% do everything in log-space for numerical reasons....
    Pc_d_pos_test = [];
    %%% positive model on positive training images
    for a=1:size(te_out_files,1)
        Pc_d_pos_test(1,a) = log(class_priors(1)) + sum(X_fg(:,a) .* log(Pw_pos)); 
    end

    %%% negative model on positive training images
    for a=1:size(te_out_files,1)
        Pc_d_pos_test(2,a) = log(class_priors(2)) + sum(X_fg(:,a) .* log(Pw_neg)); 
    end
    Pc_d_neg_test = [];
    %%% positive model on negative training images
    for a=1:size(te_in_files,1)
        Pc_d_neg_test(1,a) = log(class_priors(1)) + sum(X_bg(:,a) .* log(Pw_pos)); 
    end

    %%% negative model on negitive training images
    for a=1:size(te_in_files,1)
        Pc_d_neg_test(2,a) = log(class_priors(2)) + sum(X_bg(:,a) .* log(Pw_neg)); 
    end
    values = [Pc_d_pos_test(1,:)-Pc_d_pos_test(2,:) , Pc_d_neg_test(1,:)-Pc_d_neg_test(2,:)];
    labels = [ones(1,length(te_out_files)) , zeros(1,length(te_in_files))];

    %%% compute roc
    [roc_curve_test,roc_op_test,roc_area_test,roc_threshold_test] = roc([values;labels]');
    predict_inds = values > roc_threshold_test;
    tmp_te_split = te_split{i};
    tmp_te_predict = tmp_te_split(predict_inds);
    pred_outdoor(tmp_te_predict) = 1;
end
result_path = [result_dir filesep 'indoor_outdoor_predict.mat'];
save(result_path, 'pred_outdoor');
outdoor = double(outdoor);
correct_outdoor = outdoor' .* pred_outdoor;
outdoor_rate = sum(correct_outdoor(:))/sum(outdoor(:));
indoor = 1- outdoor;
pred_indoor = 1- pred_outdoor;
correct_indoor = indoor' .* pred_indoor;
indoor_rate = sum(correct_indoor(:))/sum(indoor(:));


result_path = [result_dir filesep 'indoor_outdoor_predict.mat'];
load(result_path, 'pred_outdoor');
outdoor = double(outdoor);
correct_outdoor = outdoor' .* pred_outdoor;
outdoor_rate = sum(correct_outdoor(:))/sum(outdoor(:))
indoor = 1- outdoor;
pred_indoor = 1- pred_outdoor;
correct_indoor = indoor' .* pred_indoor;
indoor_rate = sum(correct_indoor(:))/sum(indoor(:))