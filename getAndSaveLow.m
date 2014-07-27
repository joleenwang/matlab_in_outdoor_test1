feature_path='D:\test\1����-����\test20140726-new-feature\train_lowLevelFeature2.txt';%����������ļ�
predict_feature_path='D:\test\1����-����\test20140726-new-feature\predict_lowLevelFeature2.txt';%����������ļ�
indoor_dir = 'D:\TDDownload\ColorCheckerDatabase\indoor\';%indoorͼƬ
outdoor_dir = 'D:\TDDownload\ColorCheckerDatabase\outdoor\';%outdoorͼƬ
type='*.tif';

%feature_path='D:\test\2\����2014-7-26 172212\train_lowLevelFeature.txt';%����������ļ�
%indoor_dir = 'D:\test\2\2natural\';%indoorͼƬ
%outdoor_dir = 'D:\test\2\2urban\';%outdoorͼƬ
%type='*.jpg';

fid=fopen(feature_path,'wt');%�򿪴���������ļ�
fid2=fopen(predict_feature_path,'wt');%�򿪴���������ļ�

indoor_file_names = dir(fullfile(indoor_dir, type));
indoor_file_num=size(indoor_file_names,1);
label=1;

train_num=floor(indoor_file_num*2/3);
fprintf('%d ��indoor\n',train_num)
fprintf('��ʼ��indoor������������������')
for j=1:indoor_file_num
    fprintf('%d------%d\n',j,indoor_file_num)
    img_path=[indoor_dir indoor_file_names(j).name];
    feature=getLowLevelFeature2(img_path);
    %��¼����
    if(j<=train_num)
         fprintf(fid,'%d',label);
        for i=1:107
            fprintf(fid,'%s',' ');
            fprintf(fid,'%d',i);
            fprintf(fid,'%s',':');
            fprintf(fid,'%f',feature(i));
        end
        fprintf(fid,'%c\n','');
    else
        fprintf(fid2,'%d',label);
        for i=1:107
            fprintf(fid2,'%s',' ');
            fprintf(fid2,'%d',i);
            fprintf(fid2,'%s',':');
            fprintf(fid2,'%f',feature(i));
        end
        fprintf(fid2,'%c\n','');
    end
    
   
   % fclose(fid);
end


outdoor_file_names = dir(fullfile(outdoor_dir,type));
outdoor_file_num=size(outdoor_file_names,1);
label=-1;
train_num=floor(outdoor_file_num*2/3);
fprintf('%d ��outdoor\n',train_num)
fprintf('��ʼ��outdoor������������������')
for k=1:outdoor_file_num
    img_path=[outdoor_dir outdoor_file_names(k).name];
    feature=getLowLevelFeature2(img_path);
    %��¼����
    if(k<=train_num)
         fprintf(fid,'%d',label);
        for i=1:107
            fprintf(fid,'%s',' ');
            fprintf(fid,'%d',i);
            fprintf(fid,'%s',':');
            fprintf(fid,'%f',feature(i));
        end
        fprintf(fid,'%c\n','');
    else
        fprintf(fid2,'%d',label);
        for i=1:107
            fprintf(fid2,'%s',' ');
            fprintf(fid2,'%d',i);
            fprintf(fid2,'%s',':');
            fprintf(fid2,'%f',feature(i));
        end
        fprintf(fid2,'%c\n','');
    end

end
fprintf('����\n');
fclose(fid);
fclose(fid2);