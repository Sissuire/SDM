function [appearance, motion_content, orig_desc, dist_desc] = fn_universal_extract_kp(orig, dist, n_snippets)
n_len = min(size(orig, 3), size(dist, 3));
orig = orig(:,:,1:n_len);
dist = dist(:,:,1:n_len);

%% prepare
% n_snippets = 10;
n_frms = 18;
offset = 5;
grid = 48;
sample_percent = 0.6;

orig = orig(:,:,offset+1:end);
dist = dist(:,:,offset+1:end);

[rows, cols, n_len] = size(dist);
stride = floor((n_len - n_frms) / n_snippets);

%% appearance
frm_quality = zeros(n_len, 1);
appearance = zeros(n_snippets, 1);
for i = 1:n_len
    frm_quality(i) = fn_compute_GMSD(orig(:,:,i), dist(:,:,i));
end
for i = 1:n_snippets
    pos = (i-1)*stride;
    appearance(i) = mean(frm_quality(pos+1:pos+n_frms));
end
clear frm_quality

%% eig-based key-points detection
eigs = zeros(rows, cols, n_snippets);
for t = 1:n_snippets
    pos = (t-1) * stride + 1;
    tmp = fn_get_kpvalue(orig(:,:,pos));
    eigs(:,:,t) = tmp';
end

% center-shifted
nR = floor(rows / grid);
nC = floor(cols / grid);
r_off = floor((rows - grid*nR) / 2);
c_off = floor((cols - grid*nC) / 2);
thre_p = round(nR * nC * sample_percent);

eigs = eigs(r_off+1:r_off+nR*grid, c_off+1:c_off+nC*grid, :);
kp_flag = false(nR, nC, n_snippets);
for t = 1:n_snippets
    kpv = zeros(nR, nC);
    for i = 1:nR
        for j = 1:nC
            tmp = eigs((i-1)*grid+1:i*grid, (j-1)*grid+1:j*grid, t);
            kpv(i, j) = mean(tmp(:));
        end
    end
    
    [~, sort_idx] = sort(kpv(:), 'descend');
    candidate = false(size(kpv));
    candidate(sort_idx(1:thre_p)) = 1;
    kp_flag(:,:,t) = candidate;  
end


%% grid-based motion_velocity
nR = floor(rows / grid);
nC = floor(cols / grid);
nB = nR * nC;
% orig_desc = zeros(nR, nC, 32, n_frms, n_snippets);
orig_desc = zeros(thre_p, 32, n_frms, n_snippets);
dist_desc = orig_desc;
desc1 = zeros(n_len, nB*32);
desc2 = desc1;

for t = 1:n_len-1
    desc1(t, :) = fn_get_grid_HoF(orig(:,:,t), orig(:,:,t+1));
    desc2(t, :) = fn_get_grid_HoF(dist(:,:,t), dist(:,:,t+1));
end
desc1(n_len, :) = desc1(n_len-1, :);
desc2(n_len, :) = desc2(n_len-1, :);
for t = 1:n_snippets
    for f = 1:n_frms
        pos = (t - 1) * stride;
        desc10 = desc1(pos+f, :);
        desc20 = desc2(pos+f, :);
        desc11 = reshape(desc10', [32, nB]);
        desc21 = reshape(desc20', [32, nB]);
        
        tick = 1;
        for i = 1:nR
            for j = 1:nC
                if kp_flag(i,j,t) 
                    index = i + (j-1) * nR;
                    orig_desc(tick,:,f,t) = desc11(:,index)';
                    dist_desc(tick,:,f,t) = desc21(:,index)';
                    tick = tick + 1;
                end
            end
        end 
    end
end
clear desc1 desc2


%% grid-based motion_content
argT = 255;
orig = fn_compute_GM( orig );
dist = fn_compute_GM( dist );
chg = (2 * orig .* dist + argT) ./ (orig.^2 + dist.^2 + argT);
clear orig dist

% center-shifted
nR = floor(rows / grid);
nC = floor(cols / grid);
r_off = floor((rows - grid*nR) / 2);
c_off = floor((cols - grid*nC) / 2);

% motion_content = zeros(nR, nC, n_snippets);
motion_content = zeros(thre_p, n_snippets);
for t = 1:n_snippets
    tick = 1;
    for i = 1:nR
        for j = 1:nC
            if kp_flag(i, j, t)
                pos = (t - 1) * stride;
                pick = chg(r_off+(i-1)*grid+1:r_off+i*grid, c_off+(j-1)*grid+1:c_off+j*grid, pos+1:pos+n_frms);
                motion_content(tick, t) = std(pick(:));
                tick = tick + 1;
            end
        end
    end
end


appearance = single(appearance);
motion_content = single(motion_content);
orig_desc = single(orig_desc);
dist_desc = single(dist_desc);
end

function gm = fn_compute_GM( vid )
[dx, dy, dt] = fn_get_kernel();

gm = sqrt(convn(vid, dx, 'same').^2 + ...
    convn(vid, dy, 'same').^2 + ...
    convn(vid, dt, 'same').^2);
end

function score = fn_compute_motion_content(v1, v2)
argT = 255;
[dx, dy, dt] = fn_get_kernel();

gm1 = sqrt(convn(v1, dx, 'valid').^2 + ...
    convn(v1, dy, 'valid').^2 + ...
    convn(v1, dt, 'valid').^2);
gm2 = sqrt(convn(v2, dx, 'valid').^2 + ...
    convn(v2, dy, 'valid').^2 + ...
    convn(v2, dt, 'valid').^2);

chg = (2 * gm1 .* gm2 + argT) ./ (gm1.^2 + gm2.^2 + argT);
score = std(chg(:));
end

function [dx, dy, dt] = fn_get_kernel()
basis_1 = [1 1 1; 1 1 1; 1 1 1] / 9;
basis_2 = zeros(3);
basis_3 = -basis_1;

dx(:, 1, :) = basis_1;
dx(:, 2, :) = basis_2;
dx(:, 3, :) = basis_3;

dy(1, :, :) = basis_1;
dy(2, :, :) = basis_2;
dy(3, :, :) = basis_3;

dt = cat(3, basis_1, basis_2, basis_3);
end

function score = fn_compute_GMSD(im1, im2)
argT = 255;
dx = [1, 0, -1; 1, 0, -1; 1, 0, -1] / 3;
dy = dx';

gm1 = sqrt(conv2(im1, dx, 'valid').^2 + conv2(im1, dy, 'valid').^2);
gm2 = sqrt(conv2(im2, dx, 'valid').^2 + conv2(im2, dy, 'valid').^2);

chg = (2 * gm1 .* gm2 + argT) ./ (gm1.^2 + gm2.^2 + argT);
score = std(chg(:));
end