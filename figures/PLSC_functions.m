function PLSC_function = PLSC_functions
% Function interface
% Expose local functions through function handles.
PLSC_function.load_data = @load_data;
PLSC_function.get_3idx = @get_3idx;
PLSC_function.get_channel = @get_channel;
PLSC_function.data_detailHG_bipolar = @data_detailHG_bipolar;
PLSC_function.data_resample = @data_resample;
PLSC_function.data_hilbert = @data_hilbert;
PLSC_function.get_spacechannel = @get_spacechannel;
PLSC_function.regular_channellabel = @regular_channellabel;
PLSC_function.data_regular_forgranger = @data_regular_forgranger;

PLSC_function.plot_for3Dspace = @plot_for3Dspace;
PLSC_function.calculate_move_mean_sem = @calculate_move_mean_sem;


end





function [LFP_data_slice,LFP_data_move_slice,speak_word_table_zi,speak_move_word_table_zi,...
    abnormal_trs,abnormal_trs_move,channel_num] = load_data(which_patient)

% Bed condition without bipolar referencing
LFP_data_slice_forspace = load(['G:/wxl/',which_patient,'/results/data/realneed/LFP_data_slice_channels.mat']);
abnormal_trs = load(['G:/wxl/',which_patient,'/results/data/realneed/abnormal_trials.mat']);

channel_num = size(LFP_data_slice_forspace.LFP_data_slice_channels{1},2);

load(['G:/wxl/',which_patient,'/results/data/realneed/speak_word_table_zi.mat']);
load(['G:/wxl/',which_patient,'/results/data/realneed/speak_word_table.mat']);


% Walking condition without bipolar referencing
LFP_data_move_slice_forspace = load(['G:/wxl/',which_patient,'/results/data/realneed/move_LFP_data_slice_channels.mat']);
abnormal_trs_move = load(['G:/wxl/',which_patient,'/results/data/realneed/move_abnormal_trials.mat']);


load(['G:/wxl/',which_patient,'/results/data/realneed/move_speak_word_table_zi.mat']);
load(['G:/wxl/',which_patient,'/results/data/realneed/move_speak_word_table.mat']);


LFP_data_slice = cell(1,4);
LFP_data_move_slice = cell(1,4);

for j = 1:4

    LFP_data_slice{1,j} = LFP_data_slice_forspace.LFP_data_slice_channels{j};
    LFP_data_move_slice{1,j} = LFP_data_move_slice_forspace.LFP_data_move_slice_channels{j};

end


end


function [speak_word_table_zi_3,speak_move_word_table_zi_3,legend_3] = get_3idx(speak_word_table_zi,speak_move_word_table_zi)


idx_yi(1) = find(speak_word_table_zi.("前两个字符") == "医",1);
idx_yi(2) = find(speak_word_table_zi.("前两个字符") == "衣",1);

idx_wo(1) = find(speak_word_table_zi.("前两个字符") == "卧",1);
idx_wo(2) = find(speak_word_table_zi.("前两个字符") == "握",1);

idx_yao(1) = find(speak_word_table_zi.("前两个字符") == "药",1);
idx_yao(2) = find(speak_word_table_zi.("前两个字符") == "要",1);

speak_word_table_zi_3{1,1} = [speak_word_table_zi{idx_yi(1),3}{:};speak_word_table_zi{idx_yi(2),3}{:}];
speak_word_table_zi_3{2,1} = [speak_word_table_zi{idx_wo(1),3}{:};speak_word_table_zi{idx_wo(2),3}{:}];
speak_word_table_zi_3{3,1} = [speak_word_table_zi{idx_yao(1),3}{:};speak_word_table_zi{idx_yao(2),3}{:}];


idx_yi(1) = find(speak_move_word_table_zi.("前两个字符") == "医",1);
idx_yi(2) = find(speak_move_word_table_zi.("前两个字符") == "衣",1);

idx_wo(1) = find(speak_move_word_table_zi.("前两个字符") == "卧",1);
idx_wo(2) = find(speak_move_word_table_zi.("前两个字符") == "握",1);

idx_yao(1) = find(speak_move_word_table_zi.("前两个字符") == "药",1);
idx_yao(2) = find(speak_move_word_table_zi.("前两个字符") == "要",1);


speak_move_word_table_zi_3{1,1} = [speak_move_word_table_zi{idx_yi(1),3}{:};speak_move_word_table_zi{idx_yi(2),3}{:}];% yi
speak_move_word_table_zi_3{2,1} = [speak_move_word_table_zi{idx_wo(1),3}{:};speak_move_word_table_zi{idx_wo(2),3}{:}];% wo
speak_move_word_table_zi_3{3,1} = [speak_move_word_table_zi{idx_yao(1),3}{:};speak_move_word_table_zi{idx_yao(2),3}{:}];% yao
legend_3 = {'yi','wo','yao'};


end


function [channel_label_all,channel_label] = get_channel(which_patient,patient_idx)

LoadPath = ['G:/wxl/',which_patient,'/',patient_idx,'电极通道对应关系.xlsx'];
channel_label_all = readtable(LoadPath);
channel_label_all.Properties.VariableNames = {'Var1','Var2','Var3','Var4','Var5','Var6','Var7'}; % Rename columns using default variable names
channel_label = channel_label_all(:,[3]);
channel_label=table2array(channel_label);

end


function [LFP_data_bipolar,LFP_data_move_bipolar,channel_num_bipolar] = data_detailHG_bipolar(LFP_data_slice,LFP_data_move_slice,fs,channel_num,bandnum,bandwidth,ifbipolar)

LFP_data_bipolar_linshi = cell(1,bandnum);
LFP_data_move_bipolar_linshi = cell(1,bandnum);

% Reshape data into trial-by-channel-by-time arrays
for j = 1:4
    for m = 1:channel_num
        for i = 1:size(LFP_data_slice{1},1)

            LFP_data_bipolar_linshi{1,j}(i,m,:) = LFP_data_slice{j}{i,m};

        end

        for i = 1:size(LFP_data_move_slice{1},1)

            LFP_data_move_bipolar_linshi{1,j}(i,m,:) = LFP_data_move_slice{j}{i,m};

        end
    end
end

% Subdivide the high-gamma band into narrower frequency bands
LFP_data_highgamma = LFP_data_bipolar_linshi{4};
LFP_data_move_highgamma = LFP_data_move_bipolar_linshi{4};

LFP_data_highgamma_linshi = cell(1,9);
LFP_data_move_highgamma_linshi = cell(1,9);

% Apply band-pass filtering channel by channel
for j = 1:9

    f1 = bandwidth(2*(3+j)-1);
    f2 = bandwidth(2*(3+j));

    for m = 1:channel_num

        input_hg = squeeze(LFP_data_highgamma(:,m,:))';
        LFP_data_highgamma_linshi{j}(:,m,:) = (bandpass(input_hg,[f1,f2],fs))';

        input_hg = squeeze(LFP_data_move_highgamma(:,m,:))';
        LFP_data_move_highgamma_linshi{j}(:,m,:) = (bandpass(input_hg,[f1,f2],fs))';

    end
end


% Apply bipolar referencing


LFP_data_bipolar = cell(1,bandnum);
LFP_data_move_bipolar = cell(1,bandnum);


for i = 4:bandnum

    LFP_data_bipolar_linshi{i} = LFP_data_highgamma_linshi{i-3};
    LFP_data_move_bipolar_linshi{i} = LFP_data_move_highgamma_linshi{i-3};

end

for i = 1:bandnum
    for m = 1:channel_num-1

        LFP_data_bipolar{i}(:,m,:) = ( LFP_data_bipolar_linshi{i}(:,m,:) - LFP_data_bipolar_linshi{i}(:,m+1,:) );
        LFP_data_move_bipolar{i}(:,m,:) = ( LFP_data_move_bipolar_linshi{i}(:,m,:) - LFP_data_move_bipolar_linshi{i}(:,m+1,:) );

    end
end

channel_num_bipolar = channel_num - 1;


end



function [LFP_data_bipolar_resample,LFP_data_move_bipolar_resample] = data_resample(LFP_data_bipolar,LFP_data_move_bipolar,fs,fs_re,channel_num_bipolar,bandnum)

if fs_re ~= fs

    LFP_data_bipolar_resample = cell(1,bandnum);
    LFP_data_move_bipolar_resample = cell(1,bandnum);

    % Resample data channel by channel
    for j = 1:bandnum
        for m = 1:channel_num_bipolar

            input_hg = squeeze(LFP_data_bipolar{j}(:,m,:))';
            LFP_data_bipolar_resample{j}(:,m,:) = (resample(input_hg,fs_re,fs))';

            input_hg = squeeze(LFP_data_move_bipolar{j}(:,m,:))';
            LFP_data_move_bipolar_resample{j}(:,m,:) = (resample(input_hg,fs_re,fs))';

        end
    end

else

    LFP_data_bipolar_resample = LFP_data_bipolar;
    LFP_data_move_bipolar_resample = LFP_data_move_bipolar;

end

end



function [LFP_data_bipolar_resample_hilbert,LFP_data_move_bipolar_resample_hilbert] = data_hilbert(LFP_data_bipolar_resample,LFP_data_move_bipolar_resample,channel_num_bipolar,bandnum)

LFP_data_bipolar_resample_hilbert = cell(1,bandnum);
LFP_data_move_bipolar_resample_hilbert = cell(1,bandnum);

% Compute the Hilbert envelope channel by channel
for j = 1:bandnum
    for m = 1:channel_num_bipolar

        input_hg = squeeze(LFP_data_bipolar_resample{j}(:,m,:))';
        LFP_data_bipolar_resample_hilbert{j}(:,m,:) = abs((hilbert(input_hg))');

        input_hg = squeeze(LFP_data_move_bipolar_resample{j}(:,m,:))';
        LFP_data_move_bipolar_resample_hilbert{j}(:,m,:) = abs((hilbert(input_hg))');

    end
end

end



function channels_forspace = get_spacechannel( ...
    n, channel_label_all, idxxs)

%% =========================================================
% Read selected channels from functional-stability CSV files
%
% Rules:
%   1. If both files exist, read both.
%   2. If only one file exists, read the available file.
%   3. If neither file exists, raise an error.
%% =========================================================

%% Determine the number of rows to read

n = n + 1;  % Add one row because the first CSV row is the header


%% File paths

filePath = 'G:\wxl\DICE_stability_DL\DICE_stability_DL';

fileNames = cell(2,1);

fileNames{1} = [
    'detection_SEEG', ...
    char(idxxs), ...
    '_functional_stability.csv'];

fileNames{2} = [
    'decoding_SEEG', ...
    char(idxxs), ...
    '_functional_stability.csv'];

files = fullfile(filePath, fileNames);


%% Check file availability

fileExistMask = isfile(files);

if ~any(fileExistMask)

    error( ...
        ['未找到功能稳定性文件：\n', ...
        '%s\n%s'], ...
        files{1}, ...
        files{2});

end

% Keep only files that actually exist
files = files(fileExistMask);

fprintf( ...
    'Found %d functional stability file(s):\n', ...
    numel(files));

for f = 1:numel(files)

    fprintf('  %s\n', files{f});

end


%% Build channel mapping: prefix_number -> Excel row index

prefixMap = strtrim( ...
    string(channel_label_all{:,1}));

numMap = double( ...
    channel_label_all{:,2});

keyMap = prefixMap + "_" + string(numMap);

[uniqKeyMap, ia] = unique( ...
    keyMap, ...
    'stable');

rowIndexMap = ia;

key2row = containers.Map( ...
    cellstr(uniqKeyMap), ...
    num2cell(rowIndexMap));


%% Read each file and match channel labels

allRowIdx = cell(numel(files),1);

for f = 1:numel(files)

    currentFile = files{f};

    A = readcell(currentFile);

    if isempty(A)

        warning( ...
            '文件为空：%s', ...
            currentFile);

        allRowIdx{f} = [];
        continue

    end

    col1 = A(:,1);

    % Read only the first n rows while handling shorter files
    nUse = min(n, numel(col1));

    % Treat the first row as the header
    if nUse < 2

        warning( ...
            '文件没有有效数据行：%s', ...
            currentFile);

        allRowIdx{f} = [];
        continue

    end

    col1 = col1(2:nUse);

    % Convert entries to strings and trim surrounding spaces
    col1 = strtrim(string(col1));

    % Remove empty and missing entries
    validRows = ...
        col1 ~= "" & ...
        ~ismissing(col1);

    col1 = col1(validRows);

    if isempty(col1)

        warning( ...
            '文件前%d行内没有有效通道：%s', ...
            nUse, ...
            currentFile);

        allRowIdx{f} = [];
        continue

    end

    %% Extract the left contact from each bipolar channel label

    leftChan = extractBefore(col1, "-");

    noDashMask = leftChan == "";

    % If no "-" is present, use the full channel label
    leftChan(noDashMask) = col1(noDashMask);

    % Remove all spaces, e.g., po10 - po11
    leftChan = strrep(leftChan, " ", "");


    %% Split each channel label into prefix and numeric index

    tok = regexp( ...
        leftChan, ...
        '^([A-Za-z]+)(\d+)$', ...
        'tokens', ...
        'once');

    isOK = ~cellfun(@isempty, tok);

    prefix = strings(size(leftChan));
    num = nan(size(leftChan));

    prefix(isOK) = string( ...
        cellfun( ...
        @(x) x{1}, ...
        tok(isOK), ...
        'UniformOutput', false));

    num(isOK) = double(string( ...
        cellfun( ...
        @(x) x{2}, ...
        tok(isOK), ...
        'UniformOutput', false)));

    key = prefix + "_" + string(num);


    %% Map channels to row indices in channel_label_all

    rowIdx = nan(size(key));

    for i = 1:numel(key)

        if isOK(i) && ...
                isKey(key2row, char(key(i)))

            rowIdx(i) = ...
                key2row(char(key(i)));

        end

    end

    allRowIdx{f} = rowIdx;


    %% Report unmatched channels

    unmatchedMask = isnan(rowIdx);

    if any(unmatchedMask)

        fprintf( ...
            ['File %s: %d unmatched entries ', ...
            'within first %d rows.\n'], ...
            currentFile, ...
            sum(unmatchedMask), ...
            nUse);

        disp(leftChan(unmatchedMask));

    end

end


%% Merge results from all available files

nonEmptyMask = ...
    ~cellfun(@isempty, allRowIdx);

if ~any(nonEmptyMask)

    error('存在的文件中没有读取到有效通道。');

end

rowIdxAll = vertcat( ...
    allRowIdx{nonEmptyMask});

rowIdxAll = ...
    rowIdxAll(~isnan(rowIdxAll));

if isempty(rowIdxAll)

    error('所有读取到的通道均未能匹配channel_label_all。');

end

% Remove duplicates and sort in ascending order
channel_forspace = unique(rowIdxAll);


%% Retain only channels that can form adjacent bipolar pairs

channels_forspace = [];

nChannelRows = height(channel_label_all);

for i = reshape(channel_forspace,1,[])

    % Prevent i+1 from exceeding the table bounds
    if i >= nChannelRows
        continue
    end

    electrodeName1 = string( ...
        channel_label_all{i,1});

    electrodeName2 = string( ...
        channel_label_all{i+1,1});

    % Keep the pair only if adjacent contacts belong to the same electrode
    if strcmp(electrodeName1,electrodeName2)

        channels_forspace(end+1,1) = i; %#ok<AGROW>

    end

end


%% Report matching results

fprintf( ...
    'Matched channel rows before bipolar check: %d\n', ...
    numel(channel_forspace));

fprintf( ...
    'Final bipolar channels: %d\n', ...
    numel(channels_forspace));

end



function [channel_label_regular] = regular_channellabel(channel_label)


T = channel_label;
newLabel = strings(height(T),1);

for i = 1:height(T)

    str = T{i};

    % Extract all percentage values
    percent_num = regexp(str,'(\d+\.?\d*)%','tokens');
    percent_val = cellfun(@(x) str2double(x{1}), percent_num);

    % Extract all region labels associated with percentage values
    region_text = regexp(str,'\d+\.?\d*%\s*([^；;]+)','tokens');

    if ~isempty(percent_val)

        % Find the index with the highest percentage
        [~,idx] = max(percent_val);

        if idx <= length(region_text)
            newLabel(i) = string(strtrim(region_text{idx}{1}));
        else
            newLabel(i) = string(region_text{1}{1});
        end

    else
        newLabel(i) = string(str);
    end

end

% Update channel labels
channel_label_regular = (newLabel);




% Further standardize region labels and merge subregional variants

regions = string(channel_label_regular);

% Apply the same label-standardization procedure
regions_std = lower(regions);

regions_std = erase(regions_std, "_l");
regions_std = erase(regions_std, "_r");

patterns_to_remove = [
    ", anterior division"
    ", posterior division"
    ", superior division"
    ", inferior division"
    ", temporooccipital part"
    ", anterior"
    ", posterior"
    ", superior"
    ", inferior"
    ];

for i = 1:length(patterns_to_remove)
    regions_std = erase(regions_std, patterns_to_remove(i));
end

regions_std = strrep(regions_std, "heschl's gyrus", "heschls gyrus");
regions_std = strrep(regions_std, "operculum", "opercular");
regions_std = strrep(regions_std, "precuneous", "precuneus");
regions_std = strtrim(regions_std);

% Return the standardized region labels
channel_label_regular = regions_std;


end


