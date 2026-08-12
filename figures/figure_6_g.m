%% =========================================================
% All-word PLSC1 regional boxplot — channel-level analysis
%
% Main features:
%   1. Regions with fewer than minChannelsForGroup channel assignments
%      are merged into an "others" category.
%   2. The "others" category is always placed at the end of the x-axis.
%   3. Significance markers are computed using:
%        signrank(y, globalMedian, 'tail','right')
%   4. The global reference median is calculated across unique original
%      bipolar channels before cross-region duplication.
%   5. The y-axis starts at 0 and its upper limit is automatically expanded
%      according to the observed data, with a minimum value of minYUpper.
%
% Note:
%   Each scatter point represents the joint-loading score of one bipolar
%   channel assigned to a given anatomical region. A cross-region bipolar
%   channel contributes once to each constituent region. Therefore, this
%   plot is intended primarily as a channel-level descriptive visualization,
%   and significance markers should be interpreted accordingly.
%% =========================================================

clear;
close all;
clc;
rng(1);

%% ======================== Parameters =============================

baseRoot = 'G:\wxl';

patients_name = [
    "20250512"
    "20250610"
    "20250620"
    "20250826"
    "20250924"
    "20251010"
    "20250519"
    ];

patients_label = [
    "Sub014"
    "Sub016"
    "Sub017"
    "Sub018"
    "Sub019"
    "Sub020"
    "Sub015"
    ];

wordTag = "word_01";

% Anatomical-region abbreviation table: column 1 contains abbreviations and column 2 contains full region names
regionTableFile = ...
    'G:\wxl\Supplementary_table_2.xls';

% true: include only subjects for whom PLSC1 was sequentially retained as significant
useOnlySignificantPLSC1 = true;

% true: include a bipolar channel only when both constituent regions are valid
requireBothRegionsValid = true;

% Merge regions with fewer than minChannelsForGroup channel assignments into "others"
minChannelsForGroup = 5;

% false: order regions according to the abbreviation table, with "others" placed last
% true: order regions by descending median loading, with "others" placed last
sortRegionsByMedian = true;

% true: append the number of channel assignments to each x-axis label, e.g., PreC (n=18)
showChannelCountInLabel = false;

% Minimum y-axis upper limit; automatically increased when the observed data exceed this value
minYUpper = 0.16;

saveEditableFig = true;

outputFolder = fullfile( ...
    baseRoot, ...
    'PLSC_group_allword', ...
    'boxplot_channel_level');

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

%% ======================== Load anatomical-region abbreviation table =======================

if ~isfile(regionTableFile)
    error('找不到脑区简称表：%s',regionTableFile);
end

regionTable = readtable( ...
    regionTableFile, ...
    'VariableNamingRule','preserve');

if width(regionTable) < 2
    error('Supplementary_table_2.xls 至少应包含两列：简写、全称。');
end

abbrMap = strtrim(string(regionTable{:,1}));
fullMap = normalizeRegionVector(string(regionTable{:,2}));

validMap = ...
    abbrMap ~= "" & ...
    fullMap ~= "" & ...
    ~ismissing(abbrMap) & ...
    ~ismissing(fullMap);

abbrMap = abbrMap(validMap);
fullMap = fullMap(validMap);

% Remove duplicate full region names while retaining the first corresponding abbreviation
[fullMapUnique, firstIndex] = unique(fullMap, 'stable');
abbrMapUnique = abbrMap(firstIndex);

%% ======================== Aggregate channel-level loadings ==================

channelRows = {};

for s = 1:numel(patients_name)

    subject = patients_name(s);
    subjectLabel = patients_label(s);

    analysisFile = fullfile( ...
        baseRoot, ...
        char(subject), ...
        'PLSC2.0_allchannel', ...
        char(wordTag), ...
        'data', ...
        sprintf('%s_PLSC_analysis.mat', char(wordTag)));

    if ~isfile(analysisFile)
        warning('Missing file: %s', analysisFile);
        continue
    end

    tmp = load(analysisFile, 'PLSCAnalysis');

    if ~isfield(tmp, 'PLSCAnalysis')
        warning('%s 中不存在 PLSCAnalysis。', analysisFile);
        continue
    end

    A = tmp.PLSCAnalysis;

    %% Determine whether the subject should be included

    significantIndices = getFieldOrDefault( ...
        A, ...
        {'dimensions','significantIndices'}, ...
        []);

    plsc1Retained = any(significantIndices(:) == 1);

    if useOnlySignificantPLSC1 && ~plsc1Retained
        fprintf('%s skipped: PLSC1 not retained.\n', subject);
        continue
    end

    %% Load PLSC1 loading vectors

    loadingBed = getFieldOrDefault( ...
        A, ...
        {'neuralSpace','shared','Loading1'}, ...
        []);

    loadingWalking = getFieldOrDefault( ...
        A, ...
        {'neuralSpace','shared','Loading2'}, ...
        []);

    svdChannels = getFieldOrDefault( ...
        A, ...
        {'features','svdChannels'}, ...
        []);

    svdBands = getFieldOrDefault( ...
        A, ...
        {'features','svdBands'}, ...
        []);

    region1 = string(getFieldOrDefault( ...
        A, ...
        {'features','region1'}, ...
        strings(0,1)));

    region2 = string(getFieldOrDefault( ...
        A, ...
        {'features','region2'}, ...
        strings(0,1)));

    if isempty(loadingBed) || isempty(loadingWalking) || ...
            size(loadingBed,2) < 1 || size(loadingWalking,2) < 1
        warning('%s 缺少 PLSC1 loading。', subject);
        continue
    end

    Nch = numel(svdChannels);
    Nband = numel(svdBands);

    wBed = normalizeVectorL2(loadingBed(:,1));
    wWalking = normalizeVectorL2(loadingWalking(:,1));

    if Nch == 0 || Nband == 0 || ...
            numel(wBed) ~= Nch*Nband || ...
            numel(wWalking) ~= Nch*Nband || ...
            numel(region1) ~= Nch || ...
            numel(region2) ~= Nch
        warning('%s 的 loading 或通道标签尺寸不匹配。', subject);
        continue
    end

    %% Compute one joint-loading score for each bipolar channel

    WBed = reshape(wBed, Nch, Nband);
    WWalking = reshape(wWalking, Nch, Nband);

    channelJointLoading = sum( ...
        sqrt(abs(WBed) .* abs(WWalking)), ...
        2, ...
        'omitnan');

    %% Standardize anatomical region names

    region1 = normalizeRegionVector(region1);
    region2 = normalizeRegionVector(region2);

    %% Assign each bipolar channel to one or two anatomical regions

    subjectRows = cell(2*Nch, 1);
    rowCounter = 0;

    for c = 1:Nch

        valid1 = isValidRegion(region1(c));
        valid2 = isValidRegion(region2(c));

        if requireBothRegionsValid && ~(valid1 && valid2)
            continue
        end

        if valid1 && valid2

            if strcmpi(region1(c), region2(c))
                assignedRegions = region1(c);
            else
                assignedRegions = [
                    region1(c)
                    region2(c)
                    ];
            end

        elseif valid1

            assignedRegions = region1(c);

        elseif valid2

            assignedRegions = region2(c);

        else

            assignedRegions = strings(0,1);

        end

        for j = 1:numel(assignedRegions)

            currentRegion = assignedRegions(j);

            [found, loc] = ismember(currentRegion, fullMapUnique);

            if found
                currentAbbr = abbrMapUnique(loc);
            else
                currentAbbr = createFallbackAbbreviation(currentRegion);
                warning( ...
                    '简称表中未找到脑区：%s；临时简写为 %s。', ...
                    currentRegion, ...
                    currentAbbr);
            end

            rowCounter = rowCounter + 1;

            subjectRows{rowCounter} = table( ...
                subject, ...
                subjectLabel, ...
                svdChannels(c), ...
                c, ...
                region1(c), ...
                region2(c), ...
                currentRegion, ...
                currentAbbr, ...
                channelJointLoading(c), ...
                'VariableNames',{ ...
                'Subject', ...
                'SubjectLabel', ...
                'ChannelIndex', ...
                'ChannelPosition', ...
                'Region1', ...
                'Region2', ...
                'AssignedRegion', ...
                'RegionAbbreviation', ...
                'ChannelJointLoading'});

        end

    end

    subjectRows = subjectRows(1:rowCounter);

    if ~isempty(subjectRows)
        channelRows{end+1,1} = vertcat(subjectRows{:}); %#ok<SAGROW>
    end

end

if isempty(channelRows)
    error('没有得到可用于通道级箱线图的数据。');
end

ChannelRegionData = vertcat(channelRows{:});

%% ======================== Compute the global reference median =============
% Compute the median across unique original bipolar channels to avoid duplicate counting of cross-region channels

[~, uniqueChannelIndex] = unique( ...
    ChannelRegionData(:, {'Subject','ChannelIndex'}), ...
    'rows', ...
    'stable');

globalMedian = median( ...
    ChannelRegionData.ChannelJointLoading(uniqueChannelIndex), ...
    'omitnan');

%% ======================== Summarize channel-level values by region =========================

[gid, regionName, regionAbbr] = findgroups( ...
    ChannelRegionData.AssignedRegion, ...
    ChannelRegionData.RegionAbbreviation);

nChannelAssignments = splitapply( ...
    @numel, ...
    ChannelRegionData.ChannelJointLoading, ...
    gid);

nSubjects = splitapply( ...
    @(x) numel(unique(x)), ...
    ChannelRegionData.Subject, ...
    gid);

medianLoading = splitapply( ...
    @(x) median(x, 'omitnan'), ...
    ChannelRegionData.ChannelJointLoading, ...
    gid);

RegionSummaryAll = table( ...
    regionName, ...
    regionAbbr, ...
    nChannelAssignments, ...
    nSubjects, ...
    medianLoading, ...
    'VariableNames',{ ...
    'Region', ...
    'Abbreviation', ...
    'NChannelAssignments', ...
    'NSubjects', ...
    'MedianChannelJointLoading'});

%% ======================== Merge sparsely sampled regions into "others" ==============

ChannelRegionPlot = ChannelRegionData;

smallRegionMask = RegionSummaryAll.NChannelAssignments < minChannelsForGroup;
smallRegions = RegionSummaryAll.Region(smallRegionMask);

if ~isempty(smallRegions)

    mergeMask = ismember(ChannelRegionPlot.AssignedRegion, smallRegions);

    ChannelRegionPlot.AssignedRegion(mergeMask) = "others";
    ChannelRegionPlot.RegionAbbreviation(mergeMask) = "others";

end

%% Recompute regional summaries after merging

[gid2, regionName2, regionAbbr2] = findgroups( ...
    ChannelRegionPlot.AssignedRegion, ...
    ChannelRegionPlot.RegionAbbreviation);

nChannelAssignments2 = splitapply( ...
    @numel, ...
    ChannelRegionPlot.ChannelJointLoading, ...
    gid2);

nSubjects2 = splitapply( ...
    @(x) numel(unique(x)), ...
    ChannelRegionPlot.Subject, ...
    gid2);

medianLoading2 = splitapply( ...
    @(x) median(x, 'omitnan'), ...
    ChannelRegionPlot.ChannelJointLoading, ...
    gid2);

RegionSummary = table( ...
    regionName2, ...
    regionAbbr2, ...
    nChannelAssignments2, ...
    nSubjects2, ...
    medianLoading2, ...
    'VariableNames',{ ...
    'Region', ...
    'Abbreviation', ...
    'NChannelAssignments', ...
    'NSubjects', ...
    'MedianChannelJointLoading'});

%% ======================== Define x-axis region order =========================
% Always place the "others" category last

isOthers = RegionSummary.Region == "others";

RegionSummaryMain = RegionSummary(~isOthers,:);
RegionSummaryOthers = RegionSummary(isOthers,:);

if sortRegionsByMedian

    RegionSummaryMain = sortrows( ...
        RegionSummaryMain, ...
        {'MedianChannelJointLoading','NChannelAssignments'}, ...
        {'descend','descend'});

else

    tableOrder = nan(height(RegionSummaryMain),1);

    for r = 1:height(RegionSummaryMain)

        idx = find( ...
            abbrMapUnique == RegionSummaryMain.Abbreviation(r), ...
            1, ...
            'first');

        if isempty(idx)
            tableOrder(r) = inf;
        else
            tableOrder(r) = idx;
        end

    end

    RegionSummaryMain.TableOrder = tableOrder;
    RegionSummaryMain = sortrows(RegionSummaryMain, 'TableOrder');

end

RegionSummary = [RegionSummaryMain; RegionSummaryOthers];

regionOrder = RegionSummary.Region;
abbreviationOrder = RegionSummary.Abbreviation;

%% ======================== Construct numeric group indices =======================

groupIndex = nan(height(ChannelRegionPlot),1);

for r = 1:numel(regionOrder)

    groupIndex( ...
        ChannelRegionPlot.AssignedRegion == regionOrder(r)) = r;

end

%% ======================== Channel-level significance testing =======================

pValue = nan(numel(regionOrder),1);
starText = strings(numel(regionOrder),1);

for r = 1:numel(regionOrder)

    y = ChannelRegionPlot.ChannelJointLoading( ...
        ChannelRegionPlot.AssignedRegion == regionOrder(r));

    y = y(isfinite(y));

    if numel(y) >= 3
        pValue(r) = signrank( ...
            y, ...
            globalMedian, ...
            'tail', ...
            'right');
    end

    if isfinite(pValue(r))

        if pValue(r) < 0.001
            starText(r) = "***";
        elseif pValue(r) < 0.01
            starText(r) = "**";
        elseif pValue(r) < 0.05
            starText(r) = "*";
        elseif pValue(r) < 0.1
            starText(r) = ".";
        else
            starText(r) = "";
        end

    else

        starText(r) = "";

    end

end

%% ======================== Plot regional boxplots =======================

nRegions = numel(regionOrder);
figureWidth = max(1000, 180+85*nRegions);

fig = figure( ...
    'Color', 'w', ...
    'Units', 'pixels', ...
    'Position', [100 100 figureWidth 780]);

hold on;

boxplot( ...
    ChannelRegionPlot.ChannelJointLoading, ...
    groupIndex, ...
    'Positions', 1:nRegions, ...
    'Widths', 0.55, ...
    'Notch', 'off', ...
    'Symbol', '', ...
    'Colors', [0 0 0]);

set(findobj(gca,'Type','Line'), 'LineWidth', 1.7);

% Each scatter point represents one bipolar-channel assignment
for r = 1:nRegions

    y = ChannelRegionPlot.ChannelJointLoading( ...
        ChannelRegionPlot.AssignedRegion == regionOrder(r));

    y = y(isfinite(y));

    x = r + (rand(numel(y),1)-0.5)*0.18;

    scatter( ...
        x, ...
        y, ...
        24, ...
        'filled', ...
        'MarkerFaceAlpha', 0.58, ...
        'MarkerEdgeColor', 'none');

end

% Plot the global reference median as a dashed horizontal line
yline( ...
    globalMedian, ...
    'k--', ...
    'LineWidth', 1.4);

if showChannelCountInLabel

    xLabels = ...
        abbreviationOrder + ...
        " (n=" + ...
        string(RegionSummary.NChannelAssignments) + ...
        ")";

else

    xLabels = abbreviationOrder;

end

xlim([0.5, nRegions+0.5]);

% Set a standardized y-axis range
maxDataValue = max(ChannelRegionPlot.ChannelJointLoading, [], 'omitnan');
yUpper = max(minYUpper, ceil((maxDataValue*1.18)/0.02)*0.02);

if ~isfinite(yUpper) || yUpper <= 0
    yUpper = minYUpper;
end

ylim([0, yUpper]);

xticks(1:nRegions);
xticklabels(xLabels);
xtickangle(90);

xlabel('Brain region');
ylabel('Cross-context joint loading');
title('Channel-level regional contribution to shared PLSC1');

ax = gca;
ax.TickLabelInterpreter = 'none';
ax.TickLength = [0 0];
ax.TickDir = 'out';
ax.FontSize = 16;
ax.LineWidth = 1.5;
ax.Box = 'off';
ax.Position = [0.08 0.27 0.89 0.65];

% Add significance markers
for r = 1:numel(regionOrder)

    if starText(r) ~= ""

        text( ...
            r, ...
            yUpper*0.97, ...
            starText(r), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 18, ...
            'FontWeight', 'bold');

    end

end

%% ======================== Save outputs =============================

baseName = fullfile( ...
    outputFolder, ...
    'Allword_PLSC1_regional_boxplot_channel_level_others_stars');

exportgraphics( ...
    fig, ...
    [baseName,'.png'], ...
    'Resolution', 300);

if saveEditableFig
    savefig(fig, [baseName,'.fig']);
end

writetable( ...
    ChannelRegionData, ...
    fullfile( ...
    outputFolder, ...
    'Allword_channel_region_loading_values_original.csv'));

writetable( ...
    ChannelRegionPlot, ...
    fullfile( ...
    outputFolder, ...
    'Allword_channel_region_loading_values_with_others.csv'));

writetable( ...
    RegionSummaryAll, ...
    fullfile( ...
    outputFolder, ...
    'Allword_channel_region_summary_before_others.csv'));

writetable( ...
    RegionSummary, ...
    fullfile( ...
    outputFolder, ...
    'Allword_channel_region_summary_displayed.csv'));

SignificanceTable = table( ...
    regionOrder, ...
    abbreviationOrder, ...
    RegionSummary.NChannelAssignments, ...
    RegionSummary.NSubjects, ...
    RegionSummary.MedianChannelJointLoading, ...
    repmat(globalMedian, numel(regionOrder), 1), ...
    pValue, ...
    starText, ...
    'VariableNames',{ ...
    'Region', ...
    'Abbreviation', ...
    'NChannelAssignments', ...
    'NSubjects', ...
    'MedianChannelJointLoading', ...
    'GlobalMedian', ...
    'PValue_vs_GlobalMedian', ...
    'Star'});

writetable( ...
    SignificanceTable, ...
    fullfile( ...
    outputFolder, ...
    'Allword_channel_region_significance.csv'));

ChannelBoxplotResults = struct;

ChannelBoxplotResults.createdAt = datetime( ...
    'now', ...
    'Format', 'yyyy-MM-dd HH:mm:ss');

ChannelBoxplotResults.parameters = struct( ...
    'wordTag', wordTag, ...
    'useOnlySignificantPLSC1', useOnlySignificantPLSC1, ...
    'requireBothRegionsValid', requireBothRegionsValid, ...
    'minChannelsForGroup', minChannelsForGroup, ...
    'sortRegionsByMedian', sortRegionsByMedian, ...
    'scoreDefinition', ...
    "sum across bands of abs(WBed).*abs(WWalking)", ...
    'globalMedianRule', ...
    "median across unique original bipolar channels before cross-region duplication", ...
    'significanceTest', ...
    "signrank(y, globalMedian, tail=right)", ...
    'bipolarAssignmentRule', ...
    "Same-region bipolar counted once; cross-region bipolar counted once in each constituent region");

ChannelBoxplotResults.ChannelRegionDataOriginal = ChannelRegionData;
ChannelBoxplotResults.ChannelRegionDataDisplayed = ChannelRegionPlot;
ChannelBoxplotResults.RegionSummaryBeforeOthers = RegionSummaryAll;
ChannelBoxplotResults.RegionSummaryDisplayed = RegionSummary;
ChannelBoxplotResults.RegionOrder = regionOrder;
ChannelBoxplotResults.AbbreviationOrder = abbreviationOrder;
ChannelBoxplotResults.GlobalMedian = globalMedian;
ChannelBoxplotResults.PValue = pValue;
ChannelBoxplotResults.Star = starText;

save( ...
    fullfile( ...
    outputFolder, ...
    'Allword_PLSC1_channel_level_boxplot_results_others_stars.mat'), ...
    'ChannelBoxplotResults', ...
    '-v7.3');

fprintf('\nChannel-level PLSC regional boxplot completed.\n');
fprintf('Included bipolar-region assignments after merging others: %d\n', ...
    height(ChannelRegionPlot));
fprintf('Global median (old-style rule): %.6f\n', globalMedian);
fprintf('Output folder: %s\n', outputFolder);


%% =========================================================
% Local helper functions
%% =========================================================

function value = getFieldOrDefault(S, fieldPath, defaultValue)

value = S;

for i = 1:numel(fieldPath)

    if isstruct(value) && isfield(value, fieldPath{i})
        value = value.(fieldPath{i});
    else
        value = defaultValue;
        return
    end

end

end


function x = normalizeVectorL2(x)

x = x(:);
n = norm(x);

if isfinite(n) && n > 0
    x = x / n;
end

end


function region = normalizeRegionVector(region)

region = string(region);
region = strtrim(region);

for i = 1:numel(region)
    region(i) = extractMainRegionLocal(region(i));
end

region = regexprep(region, "_L$", "");
region = regexprep(region, "_R$", "");

patterns = [
    ", pars opercularis"
    ", superior division"
    ", inferior division"
    ", temporooccipital part"
    ];

for k = 1:numel(patterns)
    region = erase(region, patterns(k));
end

region = strrep(region, "operculum", "opercular");
region = strrep(region, "precuneous", "precuneus");
region = strrep(region, "Heschl's gyrus", "Heschls gyrus");
region = strrep(region, "Heschl's Gyrus", "Heschls Gyrus");
region = strrep(region, "frontal_pole", "frontal pole");

region = lower(strtrim(region));

end


function mainRegion = extractMainRegionLocal(str)

str = string(str);

percentNum = regexp( ...
    str, ...
    '(\d+\.?\d*)%', ...
    'tokens');

if isempty(percentNum)
    mainRegion = strtrim(str);
    return
end

percentValue = cellfun( ...
    @(x) str2double(x{1}), ...
    percentNum);

regionText = regexp( ...
    str, ...
    '\d+\.?\d*%\s*([^；;]+)', ...
    'tokens');

if isempty(regionText)
    mainRegion = strtrim(str);
    return
end

[~, idx] = max(percentValue);

if idx <= numel(regionText)
    mainRegion = strtrim(regionText{idx}{1});
else
    mainRegion = strtrim(regionText{1}{1});
end

end


function tf = isValidRegion(region)

region = lower(strtrim(string(region)));

tf = ...
    region ~= "" & ...
    region ~= "notavailable" & ...
    region ~= "not available" & ...
    region ~= "outbrain" & ...
    region ~= "out brain" & ...
    region ~= "white matter" & ...
    region ~= "unknown";

end


function abbreviation = createFallbackAbbreviation(region)

region = strtrim(string(region));
words = split(region);
words = words(words ~= "");

if isempty(words)

    abbreviation = "UNK";

elseif numel(words) == 1

    word = words(1);
    abbreviation = upper( ...
        extractBetween( ...
        word, ...
        1, ...
        min(4, strlength(word))));

else

    abbreviation = upper( ...
        join( ...
        extractBetween(words,1,1), ...
        ""));

end

end
