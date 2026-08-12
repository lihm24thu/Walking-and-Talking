%% =========================================================
% Load saved PLSCAnalysis results and visualize shared loading patterns
%
% Outputs:
%   1. Bed-condition absolute loading heatmap
%   2. Walking-condition absolute loading heatmap
%   3. Shared cross-context loading heatmap
%   4. Bed-versus-Walking loading scatter plot
%
% Input:
%   G:\wxl\<which_patient>\PLSC2.0_allchannel\<wordTag>\data\
%   <wordTag>_PLSC_analysis.mat
%
% Anatomical region abbreviations:
%   Supplementary_table_2.xls
%   Column 1: region abbreviation
%   Column 2: full anatomical region name
%
% Loading visualization:
%   - Bed and Walking loading vectors are L2-normalized separately.
%   - Absolute loading magnitudes are visualized.
%   - Bed and Walking heatmaps use the same fixed color scale.
%   - Shared loading can be visualized using either:
%       "geometricMean": sqrt(abs(WBed).*abs(WWalking))
%       "product"      : abs(WBed).*abs(WWalking)
%
% Scatter plot:
%   Each point represents one channel-frequency feature.
%   Cross-region bipolar features are duplicated only for regional
%   visualization and summary, while their original loading values remain unchanged.
%% =========================================================

clear;
close all;
clc;

which_patients = ["20250512";"20250610";"20250620";"20250826";"20250924";"20251010";"20250519"];

%%

% Number of subjects

for i = 1:1

    plot_save_PLSC_bed_walking_shared_heatmap_scatter_single_regions(which_patients{i})

end

%%



function plot_save_PLSC_bed_walking_shared_heatmap_scatter_single_regions(which_patient)

%% ======================== Parameters =======================

% Example subject ID
wordTag = "word_01";

baseRoot = 'G:\wxl';

% Anatomical region abbreviation table
regionTableFile = ...
    'G:\wxl\Supplementary_table_2.xls';

% "significant": plot all sequentially retained significant shared dimensions
% "manual"     : plot only the dimensions specified in manualDims
plotMode = "significant";
manualDims = 1;

% Shared-loading visualization:
% "geometricMean" matches the feature-level quantity used in the
% channel-level cross-context joint-loading score.
% "product" provides an alternative visualization of loading overlap.
sharedLoadingMode = "geometricMean";

% Percentile used to determine the scatter-axis range.
% Set to 100 to use the maximum value.
colorLimitPercentile = 99;

% X-axis labeling option
showChannelNumber = false;

% Scatter-plot options
showIdentityLine = true;
scatterMarkerSize = 22;
scatterMarkerAlpha = 0.50;

% Figure display and saving options
saveEditableFig = true;
showFigures = true;
closeAfterSaving = false;

analysisFile = fullfile( ...
    baseRoot, ...
    char(which_patient), ...
    'PLSC2.0_allchannel', ...
    char(wordTag), ...
    'data', ...
    sprintf('%s_PLSC_analysis.mat',char(wordTag)));

outputFolder = fullfile( ...
    baseRoot, ...
    char(which_patient), ...
    'PLSC2.0_allchannel', ...
    char(wordTag), ...
    'figures', ...
    'loading_bed_walking_shared_scatter');

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

%% ======================== Load region abbreviation table ==========

if ~isfile(regionTableFile)
    error('Cannot find region abbreviation table: %s',regionTableFile);
end

regionTable = readtable( ...
    regionTableFile, ...
    'VariableNamingRule','preserve');

if width(regionTable) < 2
    error(['Supplementary_table_2.xls must contain at least two columns: ', ...
        'abbreviation and full region name.']);
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

[fullMapUnique,firstIndex] = unique(fullMap,'stable');
abbrMapUnique = abbrMap(firstIndex);

%% ======================== Load saved PLSC analysis result ============

if ~isfile(analysisFile)
    error('Cannot find PLSC result file: %s',analysisFile);
end

loadedData = load(analysisFile,'PLSCAnalysis');

if ~isfield(loadedData,'PLSCAnalysis')
    error('The file does not contain the variable PLSCAnalysis.');
end

A = loadedData.PLSCAnalysis;

%% ======================== Extract required analysis fields ==============

loadingBed = getFieldOrDefault( ...
    A,{'neuralSpace','shared','Loading1'},[]);

loadingWalking = getFieldOrDefault( ...
    A,{'neuralSpace','shared','Loading2'},[]);

svdChannels = getFieldOrDefault( ...
    A,{'features','svdChannels'},[]);

svdBands = getFieldOrDefault( ...
    A,{'features','svdBands'},[]);

region1 = string(getFieldOrDefault( ...
    A,{'features','region1'},strings(0,1)));

region2 = string(getFieldOrDefault( ...
    A,{'features','region2'},strings(0,1)));

significantDims = getFieldOrDefault( ...
    A,{'dimensions','significantIndices'},[]);

if isempty(loadingBed) || isempty(loadingWalking)
    warning('Shared-space loading matrices are missing.');
    return;
end

Nch = numel(svdChannels);
Nband = numel(svdBands);

if Nch == 0 || Nband == 0
    error('Channel or frequency-band information is missing.');
end

if numel(region1) ~= Nch || numel(region2) ~= Nch
    error('The number of region labels does not match the number of channels.');
end

%% ======================== Select PLSC dimensions to visualize =================

switch lower(plotMode)

    case "significant"

        plotDims = significantDims(:).';

        if isempty(plotDims)
            error('No sequentially retained significant dimensions were found.');
        end

    case "manual"

        plotDims = manualDims(:).';

    otherwise

        error('plotMode must be "significant" or "manual".');

end

maxAvailableDim = min( ...
    size(loadingBed,2), ...
    size(loadingWalking,2));

plotDims = plotDims( ...
    plotDims >= 1 & ...
    plotDims <= maxAvailableDim);

plotDims = unique(plotDims,'stable');

if isempty(plotDims)
    error('No valid PLSC dimensions are available for plotting.');
end

%% ======================== Prepare frequency-band labels ==================

bandLabels = createBandLabels(svdBands);

if numel(bandLabels) ~= Nband
    bandLabels = "Band " + string((1:Nband).');
end

%% ======================== Construct anatomical region labels ==============

region1 = normalizeRegionVector(region1);
region2 = normalizeRegionVector(region2);

abbr1 = mapRegionAbbreviation( ...
    region1,fullMapUnique,abbrMapUnique);

abbr2 = mapRegionAbbreviation( ...
    region2,fullMapUnique,abbrMapUnique);

bipolarRegionLabels = strings(Nch,1);

for c = 1:Nch

    r1 = region1(c);
    r2 = region2(c);

    a1 = abbr1(c);
    a2 = abbr2(c);

    valid1 = isValidRegionLabel(r1);
    valid2 = isValidRegionLabel(r2);

    if valid1 && valid2

        if strcmpi(r1,r2)
            bipolarRegionLabels(c) = a1;
        else
            bipolarRegionLabels(c) = a1 + "-" + a2;
        end

    elseif valid1

        bipolarRegionLabels(c) = a1;

    elseif valid2

        bipolarRegionLabels(c) = a2;

    else

        bipolarRegionLabels(c) = ...
            "Ch" + string(svdChannels(c));

    end

end

%% ======================== Sort bipolar channels by region =====================
% Sort first by region abbreviation and then by bipolar-channel index.

sortTable = table( ...
    lower(bipolarRegionLabels(:)), ...
    svdChannels(:), ...
    (1:Nch)', ...
    'VariableNames',{ ...
    'RegionLabel', ...
    'ChannelIndex', ...
    'OriginalPosition'});

sortTable = sortrows( ...
    sortTable, ...
    {'RegionLabel','ChannelIndex'}, ...
    {'ascend','ascend'});

reorderChannel = sortTable.OriginalPosition;

sortedRegionLabels = bipolarRegionLabels(reorderChannel);
sortedChannelNumbers = svdChannels(reorderChannel);

if showChannelNumber

    xTickLabels = ...
        "Ch" + string(sortedChannelNumbers(:)) + ...
        ": " + sortedRegionLabels(:);

else

    xTickLabels = sortedRegionLabels(:);

end

% Identify boundaries between adjacent anatomical region groups
regionBoundaryPositions = find( ...
    sortedRegionLabels(1:end-1) ~= ...
    sortedRegionLabels(2:end)) + 0.5;

%% ======================== Visualize each selected dimension ===============

for dimIdx = plotDims

    %% -------------------- Load and normalize loading vectors --------

    wBed = loadingBed(:,dimIdx);
    wWalking = loadingWalking(:,dimIdx);

    wBed = normalizeVectorL2(wBed);
    wWalking = normalizeVectorL2(wWalking);

    assert( ...
        numel(wBed) == Nch*Nband, ...
        'Bed loading length does not match Nch*Nband.');

    assert( ...
        numel(wWalking) == Nch*Nband, ...
        'Walking loading length does not match Nch*Nband.');

    % Feature ordering:
    % frequency band is the outer index and channel is the inner index.
    WBed = abs(reshape(wBed,Nch,Nband));
    WWalking = abs(reshape(wWalking,Nch,Nband));

    WBedSorted = WBed(reorderChannel,:);
    WWalkingSorted = WWalking(reorderChannel,:);

    % Convert to [frequency band x channel] matrices for heatmap visualization
    WBedPlot = WBedSorted.';
    WWalkingPlot = WWalkingSorted.';

    switch lower(sharedLoadingMode)

        case "geometricmean"

            WSharedPlot = ...
                sqrt(WBedSorted .* WWalkingSorted).';

            sharedColorbarLabel = ...
                'Geometric-mean shared loading';

            sharedTitle = ...
                'Overlap across bed and walking';

        case "product"

            WSharedPlot = ...
                (WBedSorted .* WWalkingSorted).';

            sharedColorbarLabel = ...
                'Joint absolute loading product';

            sharedTitle = ...
                'Joint cross-context contribution';

        otherwise

            error(['sharedLoadingMode must be ', ...
                '"geometricMean" or "product".']);

    end

    %% -------------------- Set heatmap color limits -----------------------

    % Estimate robust upper limits from the data.
    % Fixed limits are applied below to maintain a common scale across subjects.
    commonUpper = robustUpperLimit( ...
        [WBedPlot(:);WWalkingPlot(:)], ...
        colorLimitPercentile);

    sharedUpper = robustUpperLimit( ...
        WSharedPlot(:), ...
        colorLimitPercentile);

    % Use fixed heatmap color limits for direct cross-subject comparison.
    commonUpper = 0.06;
    sharedUpper = 0.06;

    %% -------------------- Plot Bed, Walking, and shared loading heatmaps ----------------

    figureWidth = min(max(1700,Nch*48),5200);
    figureHeight = 1420;

    if showFigures
        figureVisibility = 'on';
    else
        figureVisibility = 'off';
    end

    figHeat = figure( ...
        'Color','w', ...
        'Visible',figureVisibility, ...
        'Units','pixels', ...
        'Position',[80 60 figureWidth figureHeight]);

    tl = tiledlayout( ...
        figHeat, ...
        3,1, ...
        'TileSpacing','compact', ...
        'Padding','compact');

    % Bed-condition loading
    ax1 = nexttile(tl,1);

    imagesc(ax1,WBedPlot);
    clim(ax1,[0 commonUpper]);

    cb1 = colorbar(ax1);
    cb1.Label.String = 'Normalized absolute loading';

    ylabel(ax1,'Frequency band (Hz)');
    title(ax1,'Bed','FontWeight','normal');

    formatLoadingAxisWithoutXLabels( ...
        ax1,bandLabels,Nband,Nch);

    drawRegionBoundaries( ...
        ax1,regionBoundaryPositions);

    % Walking-condition loading
    ax2 = nexttile(tl,2);

    imagesc(ax2,WWalkingPlot);
    clim(ax2,[0 commonUpper]);

    cb2 = colorbar(ax2);
    cb2.Label.String = 'Normalized absolute loading';

    ylabel(ax2,'Frequency band (Hz)');
    title(ax2,'Walking','FontWeight','normal');

    formatLoadingAxisWithoutXLabels( ...
        ax2,bandLabels,Nband,Nch);

    drawRegionBoundaries( ...
        ax2,regionBoundaryPositions);

    % Shared cross-context loading
    ax3 = nexttile(tl,3);

    imagesc(ax3,WSharedPlot);
    clim(ax3,[0 sharedUpper]);

    cb3 = colorbar(ax3);
    cb3.Label.String = sharedColorbarLabel;

    xlabel(ax3,'Bipolar channel and anatomical region');
    ylabel(ax3,'Frequency band (Hz)');
    title(ax3,sharedTitle,'FontWeight','normal');

    formatLoadingAxis( ...
        ax3,xTickLabels,bandLabels,Nch,Nband);

    drawRegionBoundaries( ...
        ax3,regionBoundaryPositions);

    linkaxes([ax1 ax2 ax3],'x');

    sgtitle( ...
        tl, ...
        sprintf( ...
        '%s | %s | Shared PLSC dimension %d loading patterns', ...
        char(which_patient), ...
        char(wordTag), ...
        dimIdx), ...
        'FontSize',18, ...
        'FontWeight','bold');

    heatmapBaseName = fullfile( ...
        outputFolder, ...
        sprintf( ...
        '%s_shared_dim_%03d_bed_walking_shared_heatmap', ...
        char(wordTag), ...
        dimIdx));

    saveFigureBoth( ...
        figHeat, ...
        heatmapBaseName, ...
        saveEditableFig);

    if closeAfterSaving
        close(figHeat);
    end

    %% -------------------- Bed-versus-Walking scatter plot by anatomical region ----------------
    % Each original channel-frequency feature has one Bed loading and
    % one Walking loading value.
    %
    % Region-assignment rule:
    %   1. Same-region bipolar channel: assigned once to that region
    %   2. Cross-region bipolar channel: assigned once to each constituent region
    %   3. Only one valid endpoint: assigned once to the valid region
    %
    % Important:
    %   Cross-region features are duplicated only for region-level
    %   visualization and regional summaries. Original feature values are
    %   preserved and are not altered by this duplication.

    %% Extract original unique channel-frequency feature pairs

    uniqueScatterX = WBedSorted(:);
    uniqueScatterY = WWalkingSorted(:);

    validUniqueFeature = ...
        isfinite(uniqueScatterX) & ...
        isfinite(uniqueScatterY);

    correlationX = uniqueScatterX(validUniqueFeature);
    correlationY = uniqueScatterY(validUniqueFeature);

    if isempty(correlationX)
        warning('No valid scatter points for dimension %d.',dimIdx);
        continue
    end

    %% Reorder anatomical labels to match the sorted loading matrices

    sortedRegion1 = region1(reorderChannel);
    sortedRegion2 = region2(reorderChannel);

    sortedAbbr1 = abbr1(reorderChannel);
    sortedAbbr2 = abbr2(reorderChannel);

    %% Assign each channel-frequency feature to one or two anatomical regions

    maxAssignments = 2*Nch*Nband;

    regionScatterX = nan(maxAssignments,1);
    regionScatterY = nan(maxAssignments,1);

    regionScatterLabel = strings(maxAssignments,1);
    regionScatterChannel = nan(maxAssignments,1);
    regionScatterBand = strings(maxAssignments,1);

    % Used only to separate duplicated cross-region points visually.
    % The underlying loading values remain unchanged.
    regionScatterSide = zeros(maxAssignments,1);

    assignmentCounter = 0;

    for b = 1:Nband

        for c = 1:Nch

            xValue = WBedSorted(c,b);
            yValue = WWalkingSorted(c,b);

            if ~isfinite(xValue) || ~isfinite(yValue)
                continue
            end

            r1 = sortedRegion1(c);
            r2 = sortedRegion2(c);

            a1 = sortedAbbr1(c);
            a2 = sortedAbbr2(c);

            valid1 = isValidRegionLabel(r1);
            valid2 = isValidRegionLabel(r2);

            if valid1 && valid2

                if strcmpi(r1,r2)

                    assignedRegions = a1;
                    assignedSides = 0;

                else

                    % Assign the same feature to both constituent anatomical regions
                    assignedRegions = [
                        a1
                        a2
                        ];

                    % Apply symmetric plot-only offsets to duplicated cross-region points
                    assignedSides = [
                        -1
                        1
                        ];

                end

            elseif valid1

                assignedRegions = a1;
                assignedSides = 0;

            elseif valid2

                assignedRegions = a2;
                assignedSides = 0;

            else

                assignedRegions = strings(0,1);
                assignedSides = zeros(0,1);

            end

            for a = 1:numel(assignedRegions)

                assignmentCounter = assignmentCounter+1;

                regionScatterX(assignmentCounter) = xValue;
                regionScatterY(assignmentCounter) = yValue;

                regionScatterLabel(assignmentCounter) = ...
                    assignedRegions(a);

                regionScatterChannel(assignmentCounter) = ...
                    sortedChannelNumbers(c);

                regionScatterBand(assignmentCounter) = ...
                    bandLabels(b);

                regionScatterSide(assignmentCounter) = ...
                    assignedSides(a);

            end

        end

    end

    regionScatterX = regionScatterX(1:assignmentCounter);
    regionScatterY = regionScatterY(1:assignmentCounter);

    regionScatterLabel = ...
        regionScatterLabel(1:assignmentCounter);

    regionScatterChannel = ...
        regionScatterChannel(1:assignmentCounter);

    regionScatterBand = ...
        regionScatterBand(1:assignmentCounter);

    regionScatterSide = ...
        regionScatterSide(1:assignmentCounter);

    if isempty(regionScatterX)
        warning('No region-assigned scatter points for dimension %d.',dimIdx);
        continue
    end

    %% Determine scatter-axis range

    scatterUpper = robustUpperLimit( ...
        [correlationX;correlationY], ...
        colorLimitPercentile);

    scatterMaximum = max( ...
        [correlationX;correlationY], ...
        [], ...
        'omitnan');

    % Keep extreme points largely visible when percentile-based scaling is used
    scatterUpper = max( ...
        scatterUpper, ...
        0.90*scatterMaximum);

    if ~isfinite(scatterUpper) || scatterUpper <= 0
        scatterUpper = 1;
    end

    %% Group feature assignments by anatomical region

    [regionGroupIndex,regionNames] = ...
        findgroups(regionScatterLabel);

    nRegions = numel(regionNames);

    [regionColors,regionMarkers] = ...
        getRegionPlotStyles(regionNames);
    %% Apply plot-only jitter to duplicated cross-region assignments
    % Offset points perpendicular to y=x so both assigned-region colors remain visible.
    % Regional summary statistics use the original non-jittered coordinates.

    plotJitter = 0.0035*scatterUpper;

    scatterPlotX = ...
        regionScatterX + regionScatterSide*plotJitter;

    scatterPlotY = ...
        regionScatterY - regionScatterSide*plotJitter;

    scatterPlotX = max(scatterPlotX,0);
    scatterPlotY = max(scatterPlotY,0);

    %% Create Bed-versus-Walking scatter figure

    figScatter = figure( ...
        'Color','w', ...
        'Visible',figureVisibility, ...
        'Units','pixels', ...
        'Position',[100 60 1120 900]);

    hold on;

    % Small markers represent individual channel-frequency feature assignments
    for r = 1:nRegions

        currentMask = regionGroupIndex == r;

        scatter( ...
            scatterPlotX(currentMask), ...
            scatterPlotY(currentMask), ...
            scatterMarkerSize, ...
            regionMarkers{r}, ...
            'filled', ...
            'MarkerFaceColor',regionColors(r,:), ...
            'MarkerFaceAlpha',scatterMarkerAlpha, ...
            'MarkerEdgeColor',regionColors(r,:), ...
            'MarkerEdgeAlpha',0.65, ...
            'LineWidth',0.5, ...
            'HandleVisibility','off');

    end

    %% Plot the identity line

    if showIdentityLine

        plot( ...
            [0 scatterUpper], ...
            [0 scatterUpper], ...
            'k--', ...
            'LineWidth',1.5, ...
            'HandleVisibility','off');

    end

    %% Compute regional medians from the original non-jittered coordinates

    regionMedianBed = splitapply( ...
        @(x)median(x,'omitnan'), ...
        regionScatterX, ...
        regionGroupIndex);

    regionMedianWalking = splitapply( ...
        @(x)median(x,'omitnan'), ...
        regionScatterY, ...
        regionGroupIndex);

    regionMedianShared = splitapply( ...
        @(x,y)median(sqrt(x.*y),'omitnan'), ...
        regionScatterX, ...
        regionScatterY, ...
        regionGroupIndex);

    regionFeatureCount = splitapply( ...
        @numel, ...
        regionScatterX, ...
        regionGroupIndex);

    % Large markers indicate regional median Bed and Walking loading values
    for r = 1:nRegions

        scatter( ...
            regionMedianBed(r), ...
            regionMedianWalking(r), ...
            125, ...
            regionMarkers{r}, ...
            'filled', ...
            'MarkerFaceColor',regionColors(r,:), ...
            'MarkerEdgeColor','k', ...
            'LineWidth',1.0, ...
            'HandleVisibility','off');


    end

    %% Format scatter axes

    xlim([0 scatterUpper]);
    ylim([0 scatterUpper]);

    axis square;

    xlabel('Bed normalized absolute loading');
    ylabel('Walking normalized absolute loading');

    title(sprintf( ...
        '%s | %s | PLSC dimension %d | n = %d', ...
        char(which_patient), ...
        char(wordTag), ...
        dimIdx, ...
        numel(correlationX)));

    axScatter = gca;
    axScatter.TickDir = 'out';
    axScatter.TickLength = [0 0];
    axScatter.FontSize = 14;
    axScatter.LineWidth = 1.3;
    axScatter.Box = 'off';

    %% Create a separate legend for anatomical regions
    % The legend lists individual anatomical regions only; bipolar region
    % combinations such as PreC-PostC are not treated as separate entries.

    legendHandles = gobjects(nRegions,1);

    for r = 1:nRegions

        legendHandles(r) = scatter( ...
            nan, ...
            nan, ...
            85, ...
            regionMarkers{r}, ...
            'filled', ...
            'MarkerFaceColor',regionColors(r,:), ...
            'MarkerEdgeColor','k', ...
            'LineWidth',0.8);

    end

    lgd = legend( ...
        legendHandles, ...
        regionNames, ...
        'Location','southoutside', ...
        'Interpreter','none', ...
        'Box','off');

    lgd.NumColumns = min(5,max(2,ceil(nRegions/5)));

    % Reserve space below the axes for the multi-column legend
    axScatter.Position = [0.10 0.30 0.82 0.62];

    %% Save regional assignment and summary tables

    RegionScatterAssignments = table( ...
        regionScatterChannel, ...
        regionScatterBand, ...
        regionScatterLabel, ...
        regionScatterX, ...
        regionScatterY, ...
        regionScatterSide ~= 0, ...
        'VariableNames',{ ...
        'ChannelIndex', ...
        'FrequencyBand', ...
        'AssignedRegion', ...
        'BedAbsoluteLoading', ...
        'WalkingAbsoluteLoading', ...
        'FromCrossRegionBipolar'});

    RegionScatterSummary = table( ...
        regionNames, ...
        regionFeatureCount, ...
        regionMedianBed, ...
        regionMedianWalking, ...
        regionMedianShared, ...
        'VariableNames',{ ...
        'Region', ...
        'NFeatureAssignments', ...
        'MedianBedLoading', ...
        'MedianWalkingLoading', ...
        'MedianSharedLoading'});

    writetable( ...
        RegionScatterAssignments, ...
        fullfile( ...
        outputFolder, ...
        sprintf( ...
        '%s_shared_dim_%03d_region_scatter_assignments.csv', ...
        char(wordTag), ...
        dimIdx)));

    writetable( ...
        RegionScatterSummary, ...
        fullfile( ...
        outputFolder, ...
        sprintf( ...
        '%s_shared_dim_%03d_region_scatter_summary.csv', ...
        char(wordTag), ...
        dimIdx)));

    %% Save scatter figure

    scatterBaseName = fullfile( ...
        outputFolder, ...
        sprintf( ...
        '%s_shared_dim_%03d_bed_walking_scatter_single_regions', ...
        char(wordTag), ...
        dimIdx));

    saveFigureBoth( ...
        figScatter, ...
        scatterBaseName, ...
        saveEditableFig);

    if closeAfterSaving
        close(figScatter);
    end

    %% -------------------- Save plotted loading matrices and metadata --------------

    LoadingPlotData = struct;

    LoadingPlotData.subject = which_patient;
    LoadingPlotData.wordTag = wordTag;
    LoadingPlotData.dimension = dimIdx;
    LoadingPlotData.sharedLoadingMode = sharedLoadingMode;

    LoadingPlotData.svdChannelsOriginal = svdChannels;
    LoadingPlotData.reorderChannel = reorderChannel;
    LoadingPlotData.sortedChannels = sortedChannelNumbers;
    LoadingPlotData.sortedRegionLabels = sortedRegionLabels;
    LoadingPlotData.xTickLabels = xTickLabels;
    LoadingPlotData.bandLabels = bandLabels;

    LoadingPlotData.WBedAbsolute = WBedSorted;
    LoadingPlotData.WWalkingAbsolute = WWalkingSorted;
    LoadingPlotData.WSharedDisplay = WSharedPlot.';

    LoadingPlotData.commonColorUpper = commonUpper;
    LoadingPlotData.sharedColorUpper = sharedUpper;

    LoadingPlotData.scatterBedUniqueFeatures = correlationX;
    LoadingPlotData.scatterWalkingUniqueFeatures = correlationY;

    LoadingPlotData.RegionScatterAssignments = ...
        RegionScatterAssignments;

    LoadingPlotData.RegionScatterSummary = ...
        RegionScatterSummary;

    save( ...
        fullfile( ...
        outputFolder, ...
        sprintf( ...
        '%s_shared_dim_%03d_loading_plot_data.mat', ...
        char(wordTag), ...
        dimIdx)), ...
        'LoadingPlotData', ...
        '-v7.3');

end

fprintf('\nPLSC loading visualization completed.\n');
fprintf('Input file: %s\n',analysisFile);
fprintf('Region table: %s\n',regionTableFile);
fprintf('Output folder: %s\n',outputFolder);


end


%% =========================================================
% Local helper functions
%% =========================================================

function value = getFieldOrDefault(S,fieldPath,defaultValue)

value = S;

for i = 1:numel(fieldPath)

    fieldName = fieldPath{i};

    if isstruct(value) && isfield(value,fieldName)
        value = value.(fieldName);
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
    x = x/n;
end

end


function region = normalizeRegionVector(region)

region = string(region);
region = strtrim(region);

for i = 1:numel(region)
    region(i) = extractMainRegionLocal(region(i));
end

region = regexprep(region,"_L$","");
region = regexprep(region,"_R$","");

patterns = [
    ", pars opercularis"
    ", superior division"
    ", inferior division"
    ", temporooccipital part"
    ];

for k = 1:numel(patterns)
    region = erase(region,patterns(k));
end

region = strrep(region,"operculum","opercular");
region = strrep(region,"precuneous","precuneus");
region = strrep(region,"Heschl's gyrus","Heschls gyrus");
region = strrep(region,"Heschl's Gyrus","Heschls gyrus");
region = strrep(region,"frontal_pole","frontal pole");

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
    @(x)str2double(x{1}), ...
    percentNum);

regionText = regexp( ...
    str, ...
    '\d+\.?\d*%\s*([^；;]+)', ...
    'tokens');

if isempty(regionText)
    mainRegion = strtrim(str);
    return
end

[~,idx] = max(percentValue);

if idx <= numel(regionText)
    mainRegion = strtrim(regionText{idx}{1});
else
    mainRegion = strtrim(regionText{1}{1});
end

end


function tf = isValidRegionLabel(regionName)

regionName = lower(strtrim(string(regionName)));

tf = ...
    regionName ~= "" & ...
    regionName ~= "notavailable" & ...
    regionName ~= "not available" & ...
    regionName ~= "outbrain" & ...
    regionName ~= "out brain" & ...
    regionName ~= "white matter" & ...
    regionName ~= "unknown";

end


function abbreviation = mapRegionAbbreviation( ...
    region,fullMapUnique,abbrMapUnique)

region = string(region);
abbreviation = strings(size(region));

for i = 1:numel(region)

    currentRegion = region(i);

    if ~isValidRegionLabel(currentRegion)
        abbreviation(i) = "NA";
        continue
    end

    [found,location] = ismember( ...
        currentRegion, ...
        fullMapUnique);

    if found

        abbreviation(i) = abbrMapUnique(location);

    else

        abbreviation(i) = ...
            createFallbackAbbreviation(currentRegion);

        warning( ...
            'Region not found in abbreviation table: %s; fallback: %s.', ...
            currentRegion, ...
            abbreviation(i));

    end

end

end


function abbreviation = createFallbackAbbreviation(region)

region = strtrim(string(region));
words = split(region);
words = words(words ~= "");

if isempty(words)

    abbreviation = "UNK";

elseif numel(words) == 1

    currentWord = words(1);

    abbreviation = upper( ...
        extractBetween( ...
        currentWord, ...
        1, ...
        min(4,strlength(currentWord))));

else

    abbreviation = upper( ...
        join( ...
        extractBetween(words,1,1), ...
        ""));

end

end


function labels = createBandLabels(svdBands)

Nband = numel(svdBands);

if isstring(svdBands) || iscellstr(svdBands) || ischar(svdBands)

    labels = string(svdBands(:));

    if all(~ismissing(labels)) && all(strtrim(labels) ~= "")
        return
    end

end

if isnumeric(svdBands) && size(svdBands,2) == 2

    labels = ...
        string(svdBands(:,1)) + "-" + string(svdBands(:,2));
    return

end

if Nband == 12

    labels = [
        "1-13"
        "14-30"
        "31-60"
        "61-70"
        "71-80"
        "81-90"
        "91-100"
        "101-110"
        "111-120"
        "121-130"
        "131-140"
        "141-150"
        ];

else

    labels = "Band " + string((1:Nband).');

end

end


function upperLimit = robustUpperLimit(values,percentileValue)

values = values(:);
values = values(isfinite(values));

if isempty(values)
    upperLimit = 1;
    return
end

percentileValue = min(max(percentileValue,0),100);

if percentileValue >= 100

    upperLimit = max(values);

else

    sortedValues = sort(values,'ascend');
    n = numel(sortedValues);

    position = 1 + ...
        (n-1)*(percentileValue/100);

    lowerIndex = floor(position);
    upperIndex = ceil(position);

    if lowerIndex == upperIndex

        upperLimit = sortedValues(lowerIndex);

    else

        fraction = position-lowerIndex;

        upperLimit = ...
            sortedValues(lowerIndex)*(1-fraction) + ...
            sortedValues(upperIndex)*fraction;

    end

end

if ~isfinite(upperLimit) || upperLimit <= 0
    upperLimit = max(values);
end

if ~isfinite(upperLimit) || upperLimit <= 0
    upperLimit = 1;
end

end


function formatLoadingAxisWithoutXLabels( ...
    ax,bandLabels,Nband,Nch)

ax.XTick = [];
ax.XTickLabel = [];

ax.YTick = 1:Nband;
ax.YTickLabel = bandLabels;

ax.TickLabelInterpreter = 'none';
ax.TickLength = [0 0];
ax.TickDir = 'out';
ax.FontSize = 12;
ax.LineWidth = 1.3;
ax.Box = 'off';

xlim(ax,[0.5 Nch+0.5]);
ylim(ax,[0.5 Nband+0.5]);

end


function formatLoadingAxis( ...
    ax,xTickLabels,bandLabels,Nch,Nband)

ax.XTick = 1:Nch;
ax.XTickLabel = xTickLabels;
ax.XTickLabelRotation = 90;

ax.YTick = 1:Nband;
ax.YTickLabel = bandLabels;

ax.TickLabelInterpreter = 'none';
ax.TickLength = [0 0];
ax.TickDir = 'out';
ax.FontSize = 12;
ax.LineWidth = 1.3;
ax.Box = 'off';

xlim(ax,[0.5 Nch+0.5]);
ylim(ax,[0.5 Nband+0.5]);

end


function drawRegionBoundaries(ax,boundaryPositions)

hold(ax,'on');

for i = 1:numel(boundaryPositions)

    xline( ...
        ax, ...
        boundaryPositions(i), ...
        '-', ...
        'LineWidth',0.5, ...
        'Alpha',0.45, ...
        'HandleVisibility','off');

end

end


function saveFigureBoth(fig,baseName,saveEditableFig)

exportgraphics( ...
    fig, ...
    [baseName,'.png'], ...
    'Resolution',300);

if saveEditableFig
    savefig(fig,[baseName,'.fig']);
end

end


function [regionColors,regionMarkers] = ...
    getRegionPlotStyles(regionNames)
% Assign deterministic colors and marker shapes to anatomical regions.
%
% The same region retains the same plotting style across subjects and
% PLSC dimensions. When the number of regions exceeds the color palette,
% colors may repeat while marker shapes provide additional distinction.

regionNames = string(regionNames(:));
nRegions = numel(regionNames);

%% Fixed anatomical region order

fixedRegionOrder = [
    "Fpole"
    "IC"
    "SFG"
    "MFG"
    "IFG"
    "PreC"
    "Tpole"
    "STG"
    "MTG"
    "ITG"
    "PostC"
    "SPL"
    "SMG"
    "ANG"
    "LOC"
    "SMA"
    "Subcallosal"
    "ParaCG"
    "ACG"
    "PCG"
    "PCUN"
    "CUN"
    "PHIP"
    "LIN"
    "TFUS"
    "OFus"
    "FOper"
    "COper"
    "POper"
    "PPolare"
    "HES"
    "PTemporale"
    "THA"
    "PUT"
    "CAU"
    "PAL"
    "HIP"
    "AMYG"
    "others"
    ];

%% High-contrast categorical color palette
% RGB values are specified on a 0-255 scale and normalized to 0-1.

palette = [
    0 114 178     % blue
    213  94   0     % vermillion
    0 158 115     % green
    204 121 167     % reddish purple
    230 159   0     % orange
    86 180 233     % sky blue
    117 112 179     % purple
    27 158 119     % dark cyan
    217  95  14     % dark orange
    102 166  30     % yellow green
    231  41 138     % magenta
    166 118  29     % brown
    ]/255;

%% Marker shapes

markerSet = {
    'o'     % circle
    's'     % square
    '^'     % upward triangle
    'd'     % diamond
    'v'     % downward triangle
    '>'     % right triangle
    '<'     % left triangle
    'p'     % pentagram
    'h'     % hexagram
    };

regionColors = zeros(nRegions,3);
regionMarkers = cell(nRegions,1);

for r = 1:nRegions

    currentRegion = strtrim(regionNames(r));

    fixedIndex = find( ...
        strcmpi(currentRegion,fixedRegionOrder), ...
        1);

    % Generate a deterministic style index for regions not listed above
    if isempty(fixedIndex)

        charCode = double(char(upper(currentRegion)));

        if isempty(charCode)
            fixedIndex = 1;
        else
            fixedIndex = ...
                mod(sum(charCode)-1, ...
                numel(fixedRegionOrder))+1;
        end

    end

    colorIndex = ...
        mod(fixedIndex-1,size(palette,1))+1;

    markerIndex = ...
        mod( ...
        floor((fixedIndex-1)/size(palette,1)), ...
        numel(markerSet))+1;

    regionColors(r,:) = ...
        palette(colorIndex,:);

    regionMarkers{r} = ...
        markerSet{markerIndex};

end

end

