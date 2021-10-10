% generate features with revised FAST-VQA.
% created by Yongxu Liu (yongxu.liu@stu.xidian.edu.cn)

close all
clear
clc

db_path = 'J:/_VideoDatabase/';  % Change to your PATH of databases
database = {'LIVE', 'CSIQ', 'IVPL', 'IVC-IC'};
RofCell   = 2;
RofRegion = 48;
NofBins   = 8;

n_snippets = 10;

for i = 1
    db_name = database{i};
    
    data = load([db_path, db_name, '/', db_name, '-v.mat']);  % database info is summarized in a `.mat` file; We provide the `.mat` file in the folder './info'
    len = length(data.dst_name);
    
    for j = 1:len          % for each video
        fprintf('%d .... \t', j);
        tic
        
        ref = [db_path, db_name, '/', data.ref_name{j}];
        dst = [db_path, db_name, '/', data.dst_name{j}];
        w = data.width(j, 1); h = data.height(j, 1);
        frames = data.frameNum(j, 1);
        
        orig = funcGetScaledFrames(ref, w, h, frames);
        dist = funcGetScaledFrames(dst, w, h, frames);
        [appearance, content, orig_desc, dist_desc] = fn_universal_extract_kp(orig, dist, n_snippets);
%         [appearance, content, orig_desc, dist_desc] = fn_universal_extract_all(orig, dist, n_snippets);

        str = regexp(data.dst_name{j}, '/', 'split');
        str = str{length(str)}(1:end-4);
        
        save(['iFAST_key_', num2str(n_snippets), '/', str, '.iFAST.mat'], 'appearance', 'content', 'orig_desc', 'dist_desc', '-v7');
%         save(['iFAST_all_', num2str(n_snippets), '/', str, '.iFAST.mat'], 'appearance', 'content', 'orig_desc', 'dist_desc', '-v7');
        toc
    end
end
