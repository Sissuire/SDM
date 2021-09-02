function CompileMex()

clear
close all
clc

% set the values
opts.opencv_include_path    =   'E:\Program Files\opencv\build\include'; % OpenCV include path
opts.opencv_lib_path        =   'E:\Program Files\opencv\build\x64\vc12\lib'; % OpenCV lib path
opts.clean                  =   true; % clean mode
opts.verbose                =   1; % output verbosity

% compile flags
[cv_cflags, cv_libs] = pkg_config(opts);
mex_flags = sprintf('%s %s', cv_cflags, cv_libs);

% mex_flags = ['-v ' mex_flags];    % verbose mex output

% Compile Source File
src = 'fn_get_kpvalue.cpp Improved_TVQA.cpp Improved_TVQA_fun.cpp Improved_TVQA_def.cpp';

cmd = sprintf('mex %s %s', mex_flags, src);
eval(cmd); 

end

%
% Helper functions for windows
%
function [cflags,libs] = pkg_config(opts)
    %PKG_CONFIG  constructs OpenCV-related option flags for Windows
    I_path = opts.opencv_include_path;
    L_path = opts.opencv_lib_path;
    l_options = strcat({' -l'}, lib_names(L_path));

    l_options = [l_options{:}];

    if ~exist(I_path,'dir')
        error('OpenCV include path not found: %s', I_path);
    end
    if ~exist(L_path,'dir')
        error('OpenCV library path not found: %s', L_path);
    end

    cflags = sprintf('-I''%s''', I_path);
    libs = sprintf('-L''%s'' %s', L_path, l_options);
end

function l = lib_names(L_path)
    %LIB_NAMES  return library names
    d = dir( fullfile(L_path,'opencv_*.lib') );
    l = regexp({d.name}, '(opencv_core.+)\.lib|(opencv_video.+)\.lib|(opencv_imgproc.+)\.lib|(opencv_highgui.+)\.lib', 'tokens', 'once');
    l = [l{:}];
end