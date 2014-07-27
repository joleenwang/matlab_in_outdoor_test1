function low_level_feature=getLowLevelFeature2(img_path)
%img=imread('img\ColorCheckerDatabase\srgb8bit\8D5U5524.tif');
img=imread(img_path);
img=im2double(img);

%计算RGB，histogram，3*9=27,并且归一化
temp_R=img(:,:,1);
temp_G=img(:,:,2);
temp_B=img(:,:,3);

hist_R=hist(temp_R(:),9);
hist_G=hist(temp_G(:),9);
hist_B=hist(temp_B(:),9);

hist_R=hist_R/sum(hist_R);
hist_G=hist_G/sum(hist_G);
hist_B=hist_B/sum(hist_B);
%计算HSV,mean,3*7=21;
img_HSV=rgb2hsv(img);
horizon_length=size(img_HSV,1)/7;
mean_y=zeros(1,7);
mean_b=zeros(1,7);
mean_r=zeros(1,7);
sdev_y=zeros(1,7);
sdev_b=zeros(1,7);
sdev_r=zeros(1,7);
for i=1:7
    if(i<7)
        temp_y=img_HSV((i-1)*horizon_length+1:i*horizon_length,:,1);
        temp_b=img_HSV((i-1)*horizon_length+1:i*horizon_length,:,2);
        temp_r=img_HSV((i-1)*horizon_length+1:i*horizon_length,:,3);
    else
        temp_y=img_HSV((i-1)*horizon_length+1:end,:,1);
        temp_b=img_HSV((i-1)*horizon_length+1:end,:,2);
        temp_r=img_HSV((i-1)*horizon_length+1:end,:,3);
    end
    mean_y(i)=mean(temp_y(:));
    mean_b(i)=mean(temp_b(:));
    mean_r(i)=mean(temp_r(:));
    sdev_y(i)=std(temp_y(:));
    sdev_b(i)=std(temp_b(:));
    sdev_r(i)=std(temp_r(:));
end
%edge direction values
[gx,gy]=gaussgradient(img,0.1);
magnitude=sqrt(gx.*gx+gy.*gy);
edge_d=hist(magnitude(:),18);
edge_d=edge_d/sum(edge_d);%归一化
%coeff.of wavelet transform mean & standard deviation
img_gray=rgb2gray(img);
[c,s]=wavedec2(img_gray,3,'haar');
[s1,s2,s3]=waveGetSubGray(c,s);
coeff_mean=zeros(1,10);
coeff_std=zeros(1,10);
for i=1:10
    if(i<5)
        temp=s1(:,:,i);
        coeff_mean(i)=mean(temp(:));
        coeff_std(i)=std(temp(:));
    else
        if(i<8)
            temp=s2(:,:,i-4);
            coeff_mean(i)=mean(temp(:));
            coeff_std(i)=std(temp(:));
        else
            temp=s3(:,:,i-7);
            coeff_mean(i)=mean(temp(:));
            coeff_std(i)=std(temp(:));
        end
    end
end

low_level_feature=[hist_R hist_G hist_B mean_y mean_b mean_r sdev_y sdev_b sdev_r edge_d coeff_mean coeff_std];
