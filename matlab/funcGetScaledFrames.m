%
% funcGetScaledFrames(file, w, h, num)
%      ---- get all scaled Y frames of a given yuv420p video. All frames
%      are downscaled to lower than 256, and in most cases, this operation
%      can largely speed up and obtain a better performance.
%
% file ---- file name containing absolute address.
% w    ---- width of frame
% h    ---- height of frame
% num  ---- number of frames
%
function v = funcGetScaledFrames(file, w, h, num, scale)

if ~exist('scale', 'var')
    wh = min(w, h);
    scale = ceil(wh / 256);
end

width = ceil(w / scale);
height = ceil(h / scale);
v = zeros(height, width, num);

oper = ones(scale) / (scale*scale);

ffid = fopen(file, 'r');

for j = 1:num
    fseek(ffid, 1.5*w*h*(j-1), 'bof');
    temp = transpose(fread(ffid, [w, h], 'uint8'));
    temp = conv2(temp, oper, 'same');
    v(:,:,j) = temp(1:scale:end, 1:scale:end);
    
end
fclose(ffid);
end