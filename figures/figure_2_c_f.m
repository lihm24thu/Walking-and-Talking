%% 读取无线TXT文件
close all
clear
clc

%%

fs = 2000;
channel_num = 192;
which_patient = '';
patient_idx = '';

%% Load neural data

data_LoadPath = ['G:/wxl/',which_patient,'/',patient_idx,'.mat'];
LFP_data = load(data_LoadPath);

%% 读取保存后的EEG

%数据读取路径
[filename,path] = uigetfile(['G:\wxl\',which_patient,'\.mat']);
LoadPath = strcat(path,filename);
load(LoadPath);
raw_wired_data_all = EEG.data;
raw_wired_data_all = raw_wired_data_all';
clear EEG;

wireless_data = LFP_data;
wired_data = raw_wired_data_all;
clear raw_wired_data_all;


%% 导入有线和无线的label

%数据读取路径
[filename,path] = uigetfile(['G:\wxl\',which_patient,'\.xlsx']);
LoadPath = strcat(path,filename);
channel_label_all = readtable(LoadPath);
channel_label = channel_label_all(:,[3 5]);
channel_label = table2array(channel_label);


%% 有线无线数据对齐（单通道）
%先进行重参考
%进行滤波,手动观察确定大致范围，
%确定大致范围方法：无线数据保持不变，缩小有线数据长度，然后使用互相关法
%此部分为重参考
%此部分开启下部分
wireless_length = size(wireless_data,1);
which_channel_as_reference = 6

%无线重参考
wireless_refer_channel = channel_label(which_channel_as_reference,1);

wireless_data_refer = zeros(size(wireless_data));

for i=1:size(wireless_data,2)
    wireless_data_refer(:,i) = wireless_data(:,i) - wireless_data(:,wireless_refer_channel);
end


%有线重参考
wired_refer_channel = channel_label(which_channel_as_reference,2);

wired_data_refer = zeros(size(wired_data));

for i=1:size(wired_data,2)
    wired_data_refer(:,i) = wired_data(:,i) - wired_data(:,wired_refer_channel);
end

%% 有线无线数据对齐（单通道）
%先进行重参考
%先进行滤波，然后差分，手动观察确定大致范围，
%或者先差分，再滤波，再手动观察确定大致范围，
%确定大致范围方法：无线数据保持不变，缩小有线数据长度，然后使用互相关法
%此部分承接上部分

wired_begin = 4050000-3000000+179320-20;
wired_end = wired_begin+wireless_length-1;

%选择观察对齐无线的哪个通道
which_channel_for_align = 43;

% wireless_data_for_align=wireless_data_refer(:,channel_label(11,1));
% wired_data_for_align=wired_data_refer(wired_begin:wired_end,channel_label(8,2));


wireless_data_for_align = wireless_data_refer(:,channel_label(which_channel_for_align,1));
wired_data_for_align = wired_data_refer(wired_begin:wired_end,channel_label(which_channel_for_align,2));

wired_data_for_align = double(wired_data_for_align);


% bandpass
[b, a] = butter(4, [1, 150] / (fs / 2), 'bandpass');
wireless_data_for_align = filtfilt(b, a,wireless_data_for_align);
wired_data_for_align = filtfilt(b, a,wired_data_for_align);
% % bandstop
[b, a] = butter(4, [48, 52] / (fs / 2), 'stop');
wireless_data_for_align = filtfilt(b, a,wireless_data_for_align);
wired_data_for_align = filtfilt(b, a,wired_data_for_align);
% bandstop
[b, a] = butter(4, [98, 102] / (fs / 2), 'stop');
wireless_data_for_align = filtfilt(b, a,wireless_data_for_align);
wired_data_for_align = filtfilt(b, a,wired_data_for_align);



figure
plot(wireless_data_for_align);
hold on
plot(wired_data_for_align);
legend("wireless","wired");



%% 使用互相关法对齐(单通道)
%由于采样率不是完全的相同，因此只选择一段数据进行对齐

% 要比较的数据长度 单位为点数
howlongtoalign = 2000*100;


signal1=wireless_data_for_align(50:howlongtoalign,1);
signal2=wired_data_for_align(1:howlongtoalign+100);%冗余点用于对齐
% 计算互相关序列及相应的延迟
[c, lags] = xcorr(signal1, signal2);

% 找到互相关的最大值对应的索引
[~, idx] = max(c);

% 对应的时间延迟（单位：秒）
timeDelay = lags(idx) / fs;

fprintf('最佳对齐时的时间延迟为 %f 秒\n', timeDelay);
fprintf('最佳对齐点数为 %d\n', timeDelay*fs);

if timeDelay > 0
    % signal1 相对于 signal2 领先，向后平移 signal1
    wireless_data_refer_aligned = signal1(lags(idx)+1:end);
    wired_data_refer_aligned = signal2(1:end-lags(idx));
elseif timeDelay < 0
    % signal2 相对于 signal1 领先，向后平移 signal2
    lagAbs = abs(lags(idx));
    wireless_data_refer_aligned = signal1(1:end-lagAbs);
    wired_data_refer_aligned = signal2(lagAbs+1:end);
else
    wireless_data_refer_aligned = signal1;
    wired_data_refer_aligned = signal2;
end


% 将两段信号裁剪为一样长
wireless_data_refer_aligned = wireless_data_refer_aligned(abs(timeDelay*fs)+100:howlongtoalign-150,1);
wired_data_refer_aligned = wired_data_refer_aligned(abs(timeDelay*fs)+100:howlongtoalign-150,1);


% 可视化对齐效果
figure
plot(wireless_data_refer_aligned(1:end,1)+00,'LineWidth',1);
hold on
plot(wired_data_refer_aligned(:,1),'LineWidth',1);
legend("wireless","wired");

r = corrcoef(wireless_data_refer_aligned, wired_data_refer_aligned);  % 返回 2x2 矩阵
r_val = r(1,2);
fprintf('Pearson 相关系数 = %f\n', r_val);

% channel_label第3列为全部通道的相关系数，第5列为单通道对齐的相关系数


%% 使用互相关法对齐(全部通道)
%分两步：先滤波，再对齐
close all

channel_label(:,3) = 0;
channel_label(:,4) = 0;
wireless_data_refer_aligned_cell = cell(1,size(channel_label,1));
wired_data_refer_aligned_cell = cell(1,size(channel_label,1));


for i = 1:1:size(channel_label,1)
    which_channel_for_align = i;

    if isnan(channel_label(which_channel_for_align,2))
        continue;
    end


    wireless_data_for_align = wireless_data_refer(:,channel_label(which_channel_for_align,1));
    wired_data_for_align = wired_data_refer(wired_begin:wired_end,channel_label(which_channel_for_align,2));
    wired_data_for_align = double(wired_data_for_align);

    % bandpass
    [b, a] = butter(4, [1, 150] / (fs / 2), 'bandpass');
    wireless_data_for_align=filtfilt(b, a,wireless_data_for_align);
    wired_data_for_align=filtfilt(b, a,wired_data_for_align);
    % bandstop
    [b, a] = butter(4, [48, 52] / (fs / 2), 'stop');
    wireless_data_for_align=filtfilt(b, a,wireless_data_for_align);
    wired_data_for_align=filtfilt(b, a,wired_data_for_align);
    % % bandstop
    [b, a] = butter(4, [98, 102] / (fs / 2), 'stop');
    wireless_data_for_align=filtfilt(b, a,wireless_data_for_align);
    wired_data_for_align=filtfilt(b, a,wired_data_for_align);


    howlongtoalign = 150*2000;
    signal1 = wireless_data_for_align(50:howlongtoalign,1);
    signal2 = wired_data_for_align(1:howlongtoalign+50);%冗余点用于对齐
    % 计算互相关序列及相应的延迟
    [c, lags] = xcorr(signal1, signal2);

    % 找到互相关的最大值对应的索引
    [~, idx] = max(c);

    % 对应的时间延迟（单位：秒）
    timeDelay = lags(idx) / fs;

    fprintf('最佳对齐时的时间延迟为 %f 秒\n', timeDelay);
    fprintf('最佳对齐点数为 %d\n', timeDelay*fs);

    if abs(timeDelay*fs) > 50
        channel_label(i,4) = 1;
        continue
    end


    if timeDelay > 0
        % signal1 相对于 signal2 领先，向后平移 signal1
        wireless_data_refer_aligned = signal1(lags(idx)+1:end);
        wired_data_refer_aligned = signal2(1:end-lags(idx));
    elseif timeDelay < 0
        % signal2 相对于 signal1 领先，向后平移 signal2
        lagAbs = abs(lags(idx));
        wireless_data_refer_aligned = signal1(1:end-lagAbs);
        wired_data_refer_aligned = signal2(lagAbs+1:end);
    else
        wireless_data_refer_aligned = signal1;
        wired_data_refer_aligned = signal2;
    end


    % 将两段信号裁剪为一样长
    wireless_data_refer_aligned = wireless_data_refer_aligned(abs(timeDelay*fs)+100:howlongtoalign-100,1);
    wired_data_refer_aligned = wired_data_refer_aligned(abs(timeDelay*fs)+100:howlongtoalign-100,1);

    wireless_data_refer_aligned_cell{i} = wireless_data_refer_aligned;
    wired_data_refer_aligned_cell{i} = wired_data_refer_aligned;

    % 可视化对齐效果
    % figure(i)
    % plot(wireless_data_refer_aligned(1:end,1)+000);
    % hold on
    % plot(wired_data_refer_aligned(:,1));


    r = corrcoef(wireless_data_refer_aligned, wired_data_refer_aligned);  % 返回 2x2 矩阵
    r_val = r(1,2);
    channel_label(i,3) = r_val;
    fprintf('Pearson 相关系数 = %f\n', r_val);

end

stem(channel_label(channel_label(:,3)>0.7,3));


%% 先剔除有异常的通道

chIdx = channel_label(:,1);                % 你要检查的通道索引
X = wireless_data_refer(:, chIdx);         % [N x nCh]

amp = sqrt(mean(X.^2, 1));                 % RMS amplitude per channel (1 x nCh)
mu  = mean(amp);
sig = std(amp);

bad_global = (amp > mu + 4*sig) | (amp < mu - 4*sig);
% 硬阈值：任意时刻绝对幅值超过4000
bad_hard = any(abs(X) > 4000, 1);

bad = bad_global | bad_hard;

channel_label(:,5) = 0;
channel_label(bad,5) = 1;


%% 绘制所有时域皮尔逊相关系数

stem(channel_label(channel_label(:,3)>0.7,3));


%% 接下来是重参考之后的分析

wired_data_refer = wired_data_refer(1:wireless_length+500,:);

%% 绘制单通道时域以及psd对比

close all
which_channel_to_showpsd = 53; %33

start_idx = 80* fs;
end_idx = 100 * fs;
t = (0:(end_idx - start_idx))/fs;
x_wired = wired_data_refer_aligned_cell{which_channel_to_showpsd}(start_idx:end_idx);
x_wireless = wireless_data_refer_aligned_cell{which_channel_to_showpsd}(start_idx:end_idx);

fig2_c = figure;
plot(t,x_wireless + 1000,'Color','b','LineWidth',2);
hold on
plot(t,x_wired,'Color','r','LineWidth',2);
% legend("wireless","wired");

W = 1000; H = 600;                 % axes 绘图区尺寸（像素）
LM = 80; RM = 20;                 % 左右边距（给 ylabel / tick 留空间）
BM = 70; TM = 30;                 % 下上边距（给 xlabel / title 留空间）

fig2_c.Units = 'pixels';
fig2_c.Position = [100 100 W+LM+RM H+BM+TM];
ax = gca;
ax.FontSize = 36;
ax.TickLength = [0 0];    % x/y 的刻度线长度设为0
ylim([-600 1600]);
% xlim([5.2 7.2]);
% exportgraphics(fig2_c, "D:\data\tiantan\system3\文章手稿及图片\所有图片\fig2_wired_wireless_time_5.2-7.2_012.png");   % 300 dpi
% exportgraphics(fig2_c, "D:\data\tiantan\system3\文章手稿及图片\所有图片\fig2_wired_wireless_time_012.png");   % 300 dpi


fig2_d = figure;
pspectrum(x_wireless,fs,'FrequencyLimits',[0 150]);
hold on
pspectrum(x_wired,fs,'FrequencyLimits',[0 150]);
legend("wireless","wired");
xlabel('Frequence (Hz)');
ylabel('Power/Frequence (dB/Hz)');
grid off
title('');

fig2_d.Units = 'pixels';
fig2_d.Position = [100 100 W+LM+RM H+BM+TM];
ax = gca;
ax.FontSize = 36;
ax.TickLength = [0 0];    % x/y 的刻度线长度设为0
ylim([-22 40]);
% exportgraphics(fig2_d, "D:\data\tiantan\system3\文章手稿及图片\所有图片\fig2_wired_wireless_fre_012.png");   % 300 dpi



%% 绘制有线和无线的时频图

% which_channel_to_showpsd = 53;

close all

plot_timefre(x_wireless,t,fs,1);

plot_timefre(x_wired,t,fs,0);


function plot_timefre(x,t,fs,ifwireless)

% close all
x = zscore(x,0,1);

fb = cwtfilterbank( ...
    'SignalLength', numel(x), ...
    'SamplingFrequency', fs, ...
    'FrequencyLimits', [1 150], ...      % 频率范围
    'VoicesPerOctave', 20);              % 越大频率越细，但更慢

[wt,f] = cwt(x, 'FilterBank', fb);
P = abs(wt).^2;

Z = 10*log10(P + eps);   % log-power

fig2_e = figure;
surf(t, f, Z, 'EdgeColor', 'none');
axis tight; view(2);
set(gca,'YScale','log');
ylim([1 150]);                 % 你FrequencyLimits是[1 150]，这里也对齐一下
ylabel('Frequency (Hz)');

yticks([1 10 30 60 100 150]);
yticklabels({'1','10','30','60','100','150'});   % 可选：强制显示成整数
clim([-25 0]);
colormap(slanCM('coolwarm'))


W = 1000; H = 500;                 % axes 绘图区尺寸（像素）
LM = 80; RM = 20;                 % 左右边距（给 ylabel / tick 留空间）
BM = 70; TM = 30;                 % 下上边距（给 xlabel / title 留空间）

fig2_e.Units = 'pixels';
fig2_e.Position = [100 100 W+LM+RM H+BM+TM];
ax = gca;
ax.FontSize = 36;
ax.TickLength = [0 0];    % x/y 的刻度线长度设为0

cb = colorbar(ax, 'eastoutside');
cb.Label.String = 'Power (dB)';

% 字体大小
cb.FontSize = 20;
cb.Label.FontSize = 28;

% 刻度朝外
cb.TickDirection = 'in';


% if ifwireless
%
%     exportgraphics(fig2_e, "D:\data\tiantan\system3\文章手稿及图片\所有图片\fig2_wireless_timefre_012.png");   % 300 dpi
%
% else
%
%     exportgraphics(fig2_e, "D:\data\tiantan\system3\文章手稿及图片\所有图片\fig2_wired_timefre_012.png");   % 300 dpi
%
% end


end



%%

start_idx = 1;
end_idx = size(wireless_data_refer_aligned_cell{1},1) - 10;

% ===== User inputs =====
ch_list   = find(channel_label(:,3) > 0.9);   % 要循环的通道索引（行向量）
idx_range = start_idx:end_idx;
fs_met    = 2000;                            % 用于 Hjorth 的导数(可改成你的实际采样率)
z_thr     = 8;                              % artifact阈值

% ===== Pre-allocate results =====
nCh = numel(ch_list);

% Hjorth: [nCh x 3] -> [activity, mobility, complexity]
hj_wireless = nan(nCh,3);
hj_wired    = nan(nCh,3);

% Artifact spike count
art_wireless = nan(nCh,1);
art_wired    = nan(nCh,1);

% Kurtosis
kurt_wireless = nan(nCh,1);
kurt_wired    = nan(nCh,1);

for k = 1:nCh
    ch = ch_list(k);

    % ---- fetch aligned segments ----
    xwls = wireless_data_refer_aligned_cell{ch}(idx_range);
    xwid = wired_data_refer_aligned_cell{ch}(idx_range);

    xwls = double(xwls(:));
    xwid = double(xwid(:));

    % ---- Hjorth parameters ----
    hj_wireless(k,:) = hjorth_params(xwls, fs_met);
    hj_wired(k,:)    = hjorth_params(xwid, fs_met);

    % ---- Artifact spike count (z-score then count |z|> z_thr) ----
    zwls = (xwls - mean(xwls)) / (std(xwls) + eps);
    zwid = (xwid - mean(xwid)) / (std(xwid) + eps);

    art_wireless(k) = sum(abs(zwls) > z_thr);
    art_wired(k)    = sum(abs(zwid) > z_thr);

    % ---- Kurtosis (population kurtosis) ----
    kurt_wireless(k) = kurtosis(xwls, 1);
    kurt_wired(k)    = kurtosis(xwid, 1);
end

% ===== Pack into a table (optional) =====
T = table(ch_list(:), ...
    hj_wireless(:,1), hj_wireless(:,2), hj_wireless(:,3), ...
    art_wireless, kurt_wireless, ...
    hj_wired(:,1), hj_wired(:,2), hj_wired(:,3), ...
    art_wired, kurt_wired, ...
    'VariableNames', {'ch', ...
    'hj_act_wls','hj_mob_wls','hj_comp_wls', 'artifact_wls','kurt_wls', ...
    'hj_act_wired','hj_mob_wired','hj_comp_wired','artifact_wired','kurt_wired'});

disp(T);

% ===== helper function =====
function hj = hjorth_params(x, fs)
% Returns [activity, mobility, complexity]
x = double(x(:));

dx  = gradient(x) * fs;          % dx/dt
ddx = gradient(dx) * fs;         % d2x/dt2

activity  = var(x);
mobility  = sqrt( var(dx) / (var(x) + eps) );
mobility2 = sqrt( var(ddx) / (var(dx) + eps) );
complexity = mobility2 / (mobility + eps);

hj = [activity, mobility, complexity];
end

