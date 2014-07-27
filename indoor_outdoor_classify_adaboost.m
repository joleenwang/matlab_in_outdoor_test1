clc;
clear;
%%
addpath(genpath(fullfile('.','')));
vl_setup;
img_dir = 'img/ColorCheckerDatabase/srgb8bit/';
imdir = 'srgb8bit';
img_file_names = dir(fullfile(img_dir, '*.tif'));
num_file_names = size(img_file_names, 1);
img_names = cell(num_file_names,1);

for i = 1 : num_file_names
    img_file_name = img_file_names(i).name;
    [img_file_dir img_name img_suffix] = fileparts(img_file_name);
    img_names{i} = img_name;
end

cvfile = 'threefoldCVsplit.mat';
if ~exist(cvfile,'file')
    outdoor = load_outdoor_label(img_names);
    ind = find(outdoor==1);
    ind = ind(randperm(numel(ind))); %yue注释，就是将ind打乱顺序
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
    save(cvfile,'tr_split','te_split')
else
    load(cvfile,'te_split','tr_split');
    outdoor = load_outdoor_label(img_names);
end
if ~ exist('image_features.mat','file')
img_features = [];
tmp_files = img_names;
    for j = 1 : length(tmp_files)
        th = tic;
        tmp_file_name = tmp_files{j};
        tmp_file_path = [img_dir filesep tmp_file_name '.tif'];
        tmp_img = imread(tmp_file_path);
        [height, width, dim] = size(tmp_img);
        %% YCBCR
        YCBCR = rgb2ycbcr(tmp_img);
        gap = uint8(height /7);
        feat_per_image = zeros(107,1);
        cnt_feat = 1;
        for m = 1 : 7
            tmp1 = YCBCR( gap*(m -1)+1: gap * m, :,:);
            for n = 1 : 3
                tmp11 = tmp1(:,:,n);
                tmp11 = tmp11(:);
                tmp11 = double(tmp11);
                feat_per_image(cnt_feat) = mean(tmp11);
                cnt_feat = cnt_feat + 1;
                feat_per_image(cnt_feat) = std(tmp11);
                cnt_feat = cnt_feat + 1;
            end
        end
        %% color of histogram
        H = rgbhist(tmp_img, 3, 3,3);
        H = H(:);
        H = H';
        H = H /sum(H(:));
        feat_per_image(cnt_feat: cnt_feat + 26) = H;
        cnt_feat = cnt_feat + 27;
        %% edge of histogram
        [L a b] = RGB2Lab(tmp_img);
        sigma = 3;
        Lx=gDer(L,sigma,1,0);
        Ly=gDer(L,sigma,0,1);
        Lw=sqrt(Lx.^2+Ly.^2);
        Lx = Lx(:);
        Ly = Ly(:);
        Lw = Lw(:);
        Lx = Lx (Lw > 4);
        Ly = Ly (Lw > 4);
        if ~isempty(Lx)
            Ltheta = Ly ./(Lx + 1e-4);
            Ltheta = atan(Ltheta);
            Ltheta = (Ltheta / 3.141592653) * 180;
            theta_gap = -89 : 10 : 89;
            ntheta = histc(Ltheta, theta_gap);
            ntheta = ntheta(:);
            ntheta = ntheta';
            ntheta = ntheta/sum(ntheta(:));
            feat_per_image(cnt_feat: cnt_feat + 17) = ntheta;
        end
        cnt_feat = cnt_feat + 18;
        %% wavelet transform
        [c3,s3] = wavedec2(L,3,'db1');
        s31 = s3(1,:);
        st = 1;
        en = s31(1)*s31(2);
        ca3 = c3(st: en);
        s32 = s3(2,:);
        st = 1 + en;
        en = en + s32(1)*s32(2);
        ch3 = c3(st:en);
        st = 1 + en;
        en = en + s32(1)*s32(2);
        cv3 = c3(st:en);
        st = 1 + en;
        en = en + s32(1)*s32(2);
        cd3 = c3(st:en);
        s33 = s3(3,:);
        st = 1 + en;
        en = en + s33(1)*s33(2);
        ch2 = c3(st:en);
        st = 1 + en;
        en = en + s33(1)*s33(2);
        cv2 = c3(st:en);
        st = 1 + en;
        en = en + s33(1)*s33(2);
        cd2 = c3(st:en);
        s34 = s3(4,:);
        st = 1 + en;
        en = en + s34(1)*s34(2);
        ch1 = c3(st:en);
        st = 1 + en;
        en = en + s34(1)*s34(2);
        cv1 = c3(st:en);
        st = 1 + en;
        en = en + s34(1)*s34(2);
        cd1 = c3(st:en);
        ca3 = abs(ca3);
        feat_per_image(cnt_feat) = mean(ca3);
        cnt_feat = cnt_feat + 1;
        feat_per_image(cnt_feat) = std(ca3);
        cnt_feat = cnt_feat + 1;
        
        ch3 = abs(ch3);
        feat_per_image(cnt_feat) = mean(ch3);
        cnt_feat = cnt_feat + 1;
        feat_per_image(cnt_feat) = std(ch3);
        cnt_feat = cnt_feat + 1;
        
        cv3 = abs(cv3);
        feat_per_image(cnt_feat) = mean(cv3);
        cnt_feat = cnt_feat + 1;
        feat_per_image(cnt_feat) = std(cv3);
        cnt_feat = cnt_feat + 1;
        
        cd3 = abs(cd3);
        feat_per_image(cnt_feat) = mean(cd3);
        cnt_feat = cnt_feat + 1;
        feat_per_image(cnt_feat) = std(cd3);
        cnt_feat = cnt_feat + 1;
        
        ch2 = abs(ch2);
        feat_per_image(cnt_feat) = mean(ch2);
        cnt_feat = cnt_feat + 1;
        feat_per_image(cnt_feat) = std(ch2);
        cnt_feat = cnt_feat + 1;
        
        cv2 = abs(cv2);
        feat_per_image(cnt_feat) = mean(cv2);
        cnt_feat = cnt_feat + 1;
        feat_per_image(cnt_feat) = std(cv2);
        cnt_feat = cnt_feat + 1;
        
        cd2 = abs(cd2);
        feat_per_image(cnt_feat) = mean(cd2);
        cnt_feat = cnt_feat + 1;
        feat_per_image(cnt_feat) = std(cd2);
        cnt_feat = cnt_feat + 1;
        
        ch1 = abs(ch1);
        feat_per_image(cnt_feat) = mean(ch1);
        cnt_feat = cnt_feat + 1;
        feat_per_image(cnt_feat) = std(ch1);
        cnt_feat = cnt_feat + 1;
        
        cv1 = abs(cv1);
        feat_per_image(cnt_feat) = mean(cv1);
        cnt_feat = cnt_feat + 1;
        feat_per_image(cnt_feat) = std(cv1);
        cnt_feat = cnt_feat + 1;
        
        cd1 = abs(cd1);
        feat_per_image(cnt_feat) = mean(cd1);
        cnt_feat = cnt_feat + 1;
        feat_per_image(cnt_feat) = std(cd1);
        cnt_feat = cnt_feat + 1;
        img_features = [img_features;feat_per_image'];
        disp(sprintf('%s: %f seconds (%d/%d)\n', tmp_file_name, toc(th), j, length(tmp_files)));
    end
    save('image_features.mat','img_features');
else 
    load('image_features.mat','img_features');
end
for i=1:3 
    f = sprintf('random_forest_cv%d.mat',i);
    if ~exist(f,'file')
        data = img_features(tr_split{i},:);
        lbls = outdoor(tr_split{i});
        lbls = uint32(lbls);
        random_forest=classRF_train(data,lbls,50);
        save(f,'random_forest');
    end
end

pred_outdoor = zeros(num_file_names,1);
    for fold=1:3
        f = sprintf('random_forest_cv%d.mat',fold);
        load(f,'random_forest');
        te_ind = te_split{fold};
        data = img_features(te_ind,:);
        [Y_new, votes, prediction_per_tree] = classRF_predict(data,random_forest);
        pred_outdoor(te_ind) = Y_new;
    end
outdoor = double(outdoor);
correct_outdoor = outdoor' .* pred_outdoor;
outdoor_rate = sum(correct_outdoor(:))/sum(outdoor(:));
indoor = 1- outdoor;
pred_indoor = 1- pred_outdoor;
correct_indoor = indoor' .* pred_indoor;
indoor_rate = sum(correct_indoor(:))/sum(indoor(:));