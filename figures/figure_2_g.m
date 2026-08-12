close all
clear
clc

%% Basic information

channel_num = 192;
which_patient = '';
patient_idx = '';

%% Load neural data

data_LoadPath = ['G:/wxl/',which_patient,'/',patient_idx,'.mat'];
LFP_data = load(data_LoadPath);

%% Load channel labels

LoadPath = ['G:/wxl/',which_patient,'/',patient_idx,'电极通道对应关系.xlsx'];
channel_label_all = readtable(LoadPath);
channel_label_all.Properties.VariableNames = {'Var1','Var2','Var3','Var4','Var5','Var6'};
channel_label = channel_label_all(:,[3]);
channel_label = table2array(channel_label);

%% Time-frequency analysis of movement-related channels

close all

start_idx = 1;
fs = 2000;
end_idx = 180 * fs;

show_channels = [71,73];


%% Construct bipolar signal

x = LFP_data(start_idx:end_idx, channel_label(show_channels(1))) ...
    - LFP_data(start_idx:end_idx, channel_label(show_channels(2)));

% Standardize the bipolar signal
x = zscore(x, 0, 1);

% Construct the time axis
t = (0:numel(x)-1) / fs;


%% Continuous wavelet transform

fb = cwtfilterbank( ...
    'SignalLength', numel(x), ...
    'SamplingFrequency', fs, ...
    'FrequencyLimits', [1 150], ...
    'VoicesPerOctave', 20);

[wt, f] = cwt(x, 'FilterBank', fb);

% Calculate wavelet power
P = abs(wt).^2;

% Convert power to dB
Z = 10 * log10(P + eps);


%% Plot time-frequency representation

fig_all = figure(show_channels(1));

imagesc(t, f, Z);

% Display low frequencies at the bottom and high frequencies at the top
axis xy;

% Set color limits
% clim(cl);
clim([-14 -2]);

axis tight;

% Use a logarithmic frequency axis
set(gca, 'YScale', 'log');

ylabel('Frequency (Hz)');

yticks([1 10 30 60 100 150]);
yticklabels({'1', '10', '30', '60', '100', '150'});

% Construct the bipolar channel title
t_title = strcat( ...
    string(channel_label_all.Var1(show_channels(1))), ...
    string(channel_label_all.Var2(show_channels(1))), ...
    '-', ...
    string(channel_label_all.Var1(show_channels(2))), ...
    string(channel_label_all.Var2(show_channels(2))));

title(t_title);


%% Mark movement and stationary transition time points

% Black lines: movement onset
xline([44, 80, 126, 177], ...
    '-k', ...
    'LineWidth', 5);

% Yellow lines: stationary-state onset
xline([64, 108, 162], ...
    '-y', ...
    'LineWidth', 5);

xlim([44 177]);


%% Set colormap

% colormap(slanCM('vik'))

colormap(gca, slanCM('coolwarm'));

ax = gca;
ax.FontSize = 36;

% Remove tick marks while retaining tick labels
ax.TickLength = [0 0];
ax.XMinorTick = 'off';
ax.YMinorTick = 'off';


%% Add colorbar

ax = gca;

cb = colorbar(ax, 'eastoutside');
cb.Label.String = 'Power (dB)';

% Set font sizes
cb.FontSize = 20;
cb.Label.FontSize = 28;

% Set tick direction
cb.TickDirection = 'in';

% Reduce the number of ticks to avoid overlapping labels
% cb.Ticks = linspace(-zmax, zmax, 5);
% cb.TickLabels = compose('%.1f', cb.Ticks);

% Update figure layout
drawnow;

% Manually adjust the axes and colorbar positions
ax.Units = 'normalized';
cb.Units = 'normalized';

% Main axes position: [left bottom width height]
ax.Position = [0.06 0.18 0.84 0.72];

% Colorbar position: [left bottom width height]
% The third value controls the width of the colorbar
cb.Position = [0.915 0.18 0.020 0.72];

% Increase the distance between the colorbar label and tick labels
cb.Label.Units = 'normalized';
cb.Label.Position = [3.2 0.5 0];


%% Set figure size

W = 4000;
H = 800;

% Left and right margins
LM = 80;
RM = 180;      % Reserve space for the colorbar

% Bottom and top margins
BM = 70;
TM = 30;

fig_all.Units = 'pixels';

fig_all.Position = [ ...
    100, ...
    100, ...
    W + LM + RM, ...
    H + BM + TM];


%% Save figure

exportgraphics( ...
    fig_all, ...
    "D:\data\tiantan\system3\文章手稿及图片\所有图片\fig2_yundong_013.png", ...
    'Resolution', 300);

