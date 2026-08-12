
fs = 2000;
fs_re = 1000;

% ===== PLSC temporal processing and permutation parameters =====
output_fs = 500;       % Temporal sampling rate of the PLSC input
smooth_ms = 10;        % Gaussian smoothing window for the Hilbert envelope, in ms
minLagSec = 0.20;      % Minimum temporal offset for circular shifts, in s
sigThr = 0.95;        % 95th-percentile threshold of the null distribution

which_patient = ["20250512";"20250610";"20250620";"20250826";"20250924";"20251010";"20250519"];
patients_idx = ["0512_014";"0610_016";"0620_017";"0826_018";"0924_019";"1010_020";"0519_015"];
idxs = ["014";"016";"017";"018";"019";"020";"015"];

bandwidth = [1,13,14,30,31,60,61,70,71,80,81,90,91,100,101,110,111,120,121,130,131,140,141,150];
bandnum = length(bandwidth)/2;


%%

for nn = 1:numel(which_patient)

    PLSC_all(which_patient{nn},patients_idx{nn},idxs{nn},bandnum,fs_re, ...
        output_fs,smooth_ms,minLagSec,sigThr);

end

%%


function  PLSC_all(which_patient,patient_idx,idxxs,bandnum,fs_re, ...
    output_fs,smooth_ms,minLagSec,sigThr)

PLSC_function = PLSC_functions;

%% Load Hilbert-transformed data

load(['G:\wxl\',which_patient,'\results\data\realneed\space_LFP_data_bipolar_resample_hilbert.mat']);

load(['G:\wxl\',which_patient,'\results\data\realneed\space_LFP_data_move_bipolar_resample_hilbert.mat'],'LFP_data_move_bipolar_resample_hilbert');

load(['G:/wxl/',which_patient,'/results/data/realneed/speak_word_table_zi.mat']);

load(['G:/wxl/',which_patient,'/results/data/realneed/move_speak_word_table_zi.mat']);



%% Obtain trial labels

[speak_word_table_zi_3,speak_move_word_table_zi_3,legend_3] = PLSC_function.get_3idx(speak_word_table_zi,speak_move_word_table_zi);

legend_3

%% Obtain channel labels

[channel_label_all,channel_label] = PLSC_function.get_channel(which_patient,patient_idx);

% Standardize anatomical region labels

channel_label_all.StandardRegion = PLSC_function.regular_channellabel(channel_label_all.Var6);

%% Gaussian smoothing and resampling to the target sampling rate

LFP_data_movmean = cell(4, bandnum);
LFP_data_move_movmean = cell(4, bandnum);

if output_fs > fs_re
    error('output_fs不能高于输入采样率fs_re。');
end

window_length = max(1, round(smooth_ms / 1000 * fs_re));

% Compute the rational resampling ratio
[p_resample, q_resample] = rat(output_fs / fs_re);

fs_real = fs_re * p_resample / q_resample;

fprintf(['\nPatient %s: input fs = %.1f Hz, ', ...
    'output fs = %.1f Hz, Gaussian smoothing = %.1f ms\n'], ...
    which_patient, fs_re, fs_real, smooth_ms);

for j = 1:bandnum

    % Original data: [trial × channel × time]
    x1 = LFP_data_bipolar_resample_hilbert{1,j};
    x2 = LFP_data_move_bipolar_resample_hilbert{1,j};

    % Apply Gaussian smoothing
    x1_smooth = smoothdata(x1, 3, ...
        'gaussian', window_length);

    x2_smooth = smoothdata(x2, 3, ...
        'gaussian', window_length);

    %% Resample bed-condition data

    nTrial1 = size(x1_smooth,1);
    nCh1 = size(x1_smooth,2);
    nTime1 = size(x1_smooth,3);

    % Reshape to [time × trial*channel]
    x1_temp = reshape( ...
        permute(x1_smooth,[3,1,2]), ...
        nTime1, []);

    % Resample from fs_re to output_fs using the rational factor p/q
    x1_resampled = resample( ...
        x1_temp, p_resample, q_resample);

    nTime1New = size(x1_resampled,1);

    % Restore the data to [trial × channel × time]
    LFP_data_movmean{1,j} = permute( ...
        reshape(x1_resampled, ...
        nTime1New, nTrial1, nCh1), ...
        [2,3,1]);

    %% Resample walking-condition data

    nTrial2 = size(x2_smooth,1);
    nCh2 = size(x2_smooth,2);
    nTime2 = size(x2_smooth,3);

    x2_temp = reshape( ...
        permute(x2_smooth,[3,1,2]), ...
        nTime2, []);

    x2_resampled = resample( ...
        x2_temp, p_resample, q_resample);

    nTime2New = size(x2_resampled,1);

    LFP_data_move_movmean{1,j} = permute( ...
        reshape(x2_resampled, ...
        nTime2New, nTrial2, nCh2), ...
        [2,3,1]);

end

%% Divide trials into three word categories

for j = 1:bandnum
    for i = 1:3

        real_tr = speak_word_table_zi_3{i};
        LFP_data_movmean{i+1,j} = LFP_data_movmean{1,j}(real_tr,:,:);

        real_tr_move = speak_move_word_table_zi_3{i};
        LFP_data_move_movmean{i+1,j} = LFP_data_move_movmean{1,j}(real_tr_move,:,:);

    end
end

% The first row of LFP_data_movmean contains all trials; rows 2-4 contain the three word categories, and the 12 columns represent the 12 frequency bands.


%% Compute trial-averaged responses

LFP_data_movmean_mean = cell(4,bandnum);
LFP_data_move_movmean_mean = cell(4,bandnum);

for j = 1:bandnum
    for i = 1:4

        LFP_data_movmean_mean{i,j} = squeeze(mean(LFP_data_movmean{i,j},1));
        LFP_data_move_movmean_mean{i,j} = squeeze(mean(LFP_data_move_movmean{i,j},1));

    end
end


%% Select and combine channels ranked highly by multiple importance measures

spacechannel_num = 1000;
[channel_forspace] = PLSC_function.get_spacechannel(spacechannel_num,channel_label_all,idxxs);

additional_channel = [];

channels_forspace = union(channel_forspace,additional_channel)


%% Define the analysis time window

fprintf('PLSC实际时间采样率 fs_real = %.1f Hz\n', fs_real);


% Define the analysis time window

bed_start_time = 8.5;
bed_end_time   = 9.8;

move_start_time = 0.5;
move_end_time   = 1.8;

t_time = round(bed_start_time * fs_real) + 1 : ...
    round(bed_end_time   * fs_real);

t_time_move = round(move_start_time * fs_real) + 1 : ...
    round(move_end_time   * fs_real);

fprintf('Bed samples: %d\n', numel(t_time));
fprintf('Move samples: %d\n', numel(t_time_move));

assert(numel(t_time) == numel(t_time_move), ...
    'Bed and move analysis windows have different lengths.');


%% Construct the feature matrices for SVD/PLSC analysis
% Construct feature matrices for all four word conditions

svd_channels = channels_forspace;
svd_band = 1:12;
svd_input_mean = cell(1,4);
svd_input_move_mean = cell(1,4);

for i = 1:4
    svd_input_mean{i} = [];
    svd_input_move_mean{i} = [];
    for j = svd_band

        % Feature order: band1-ch1, band1-ch2, band1-ch3, ...
        svd_input_mean{i} = [svd_input_mean{i};LFP_data_movmean_mean{i,j}(svd_channels,t_time)];
        svd_input_move_mean{i} = [svd_input_move_mean{i};LFP_data_move_movmean_mean{i,j}(svd_channels,t_time_move)];

    end
end



%% Run the analysis for the four word conditions
space_savefolder = fullfile('G:\wxl', which_patient, 'PLSC2.0_allchannel');

if ~exist(space_savefolder, 'dir')
    mkdir(space_savefolder);
end

for diff_word = 1:4

    wordTag = sprintf('word_%02d', diff_word);
    sorted_labels = strings(0,1);

    wordSaveFolder = fullfile(space_savefolder, wordTag);
    figureFolder = fullfile(wordSaveFolder, 'figures');
    sigNullFigureFolder = fullfile(figureFolder, 'null_significant');
    nonSigNullFigureFolder = fullfile(figureFolder, 'null_nonsignificant');
    dataFolder = fullfile(wordSaveFolder, 'data');

    folderList = {
        wordSaveFolder
        figureFolder
        sigNullFigureFolder
        nonSigNullFigureFolder
        dataFolder
        };

    for iFolder = 1:numel(folderList)
        if ~exist(folderList{iFolder}, 'dir')
            mkdir(folderList{iFolder});
        end
    end

    % Prepare input matrices
    data_bed = svd_input_mean{diff_word}';
    data_move = svd_input_move_mean{diff_word}';

    % Apply feature-wise z-scoring separately within each condition
    data_bed_z = zscore(data_bed, 0, 1);
    data_move_z = zscore(data_move, 0, 1);

    if any(~isfinite(data_bed_z(:))) || any(~isfinite(data_move_z(:)))
        error('word %d的标准化数据包含NaN或Inf。', diff_word);
    end

    minLagPts = round(minLagSec * fs_real);

    % Run the circular-shift permutation test
    [numSigDims, covDist, corrDist, SigIdx, nullStats] = ...
        computeSharedNullDistribution( ...
        data_bed_z, data_move_z, minLagPts, sigThr);

    fprintf('\nWord %d: numSigDims=%d, SigIdx=', ...
        diff_word, numSigDims);
    disp(SigIdx.');

    resultStruct = getNeuralSpace( ...
        data_bed_z, data_move_z, numSigDims, SigIdx');

    nDims = size(covDist,1);

    % Dimensions that fail at least one of the two significance criteria
    PointwiseNonSigIdx = find(~nullStats.dimAll);

    % Dimensions not retained by the consecutive-dimension rule
    NonRetainedIdx = setdiff((1:nDims).', SigIdx(:), 'stable');

    % Select up to the first five pointwise non-significant dimensions for plotting
    nNonSigPlot = min(5, numel(PointwiseNonSigIdx));
    NonSigPlotIdx = PointwiseNonSigIdx(1:nNonSigPlot);

    UniIdx = (1:nDims).';
    UniIdx(ismember(UniIdx, SigIdx)) = [];

    % Save null-distribution plots for significant dimensions
    for kSig = 1:numel(SigIdx)

        dimIdx = SigIdx(kSig);

        figureBaseName = fullfile( ...
            sigNullFigureFolder, ...
            sprintf('%s_significant_dim_%03d_null', ...
            wordTag, dimIdx));

        savePLSCNullFigure( ...
            dimIdx, covDist, corrDist, nullStats, ...
            figureBaseName, 'Significant');

    end

    % Save null-distribution plots for up to five non-significant dimensions
    for kNonSig = 1:numel(NonSigPlotIdx)

        dimIdx = NonSigPlotIdx(kNonSig);

        figureBaseName = fullfile( ...
            nonSigNullFigureFolder, ...
            sprintf('%s_nonsignificant_dim_%03d_null', ...
            wordTag, dimIdx));

        savePLSCNullFigure( ...
            dimIdx, covDist, corrDist, nullStats, ...
            figureBaseName, 'Non-significant');

    end



    %% =========================================================
    %  Save shared/unique projection plots and loading maps
    %% =========================================================

    %% 1. Create output folders for figures

    sharedProjectionFolder = fullfile( ...
        figureFolder, ...
        'shared_projection');

    uniqueProjectionFolder = fullfile( ...
        figureFolder, ...
        'unique_projection');

    loadingFolder = fullfile( ...
        figureFolder, ...
        'loading');

    topFeatureFolder = fullfile( ...
        dataFolder, ...
        'top_features');

    plotFolderList = {
        sharedProjectionFolder
        uniqueProjectionFolder
        loadingFolder
        topFeatureFolder
        };

    for iFolder = 1:numel(plotFolderList)

        if ~exist(plotFolderList{iFolder}, 'dir')
            mkdir(plotFolderList{iFolder});
        end

    end


    %% 2. Construct the time axis relative to speech onset

    % Time zero denotes speech onset
    analysisStartTime = bed_start_time - 9;

    t = analysisStartTime + ...
        (0:size(data_bed_z,1)-1) / fs_real;


    %% =========================================================
    %  3. Save temporal projections of significant shared dimensions
    %% =========================================================

    if numSigDims > 0 && ...
            isfield(resultStruct, 'shared') && ...
            isfield(resultStruct.shared, 'projectedData1') && ...
            isfield(resultStruct.shared, 'projectedData2') && ...
            ~isempty(resultStruct.shared.projectedData1) && ...
            ~isempty(resultStruct.shared.projectedData2)

        nSharedPlot = min([ ...
            numSigDims, ...
            numel(SigIdx), ...
            size(resultStruct.shared.projectedData1,2), ...
            size(resultStruct.shared.projectedData2,2), ...
            30]);

        for iShared = 1:nSharedPlot

            dimIdx = SigIdx(iShared);

            zBed = ...
                resultStruct.shared.projectedData1(:,iShared);

            zMove = ...
                resultStruct.shared.projectedData2(:,iShared);

            % Handle potential mismatches between the time-axis and projection lengths
            if numel(t) == numel(zBed)

                tPlot = t;

            else

                tPlot = analysisStartTime + ...
                    (0:numel(zBed)-1) / fs_real;

            end

            rShared = corr( ...
                zBed, ...
                zMove, ...
                'Rows', 'complete');

            fprintf( ...
                'Shared dimension %d corr = %.4f\n', ...
                dimIdx, ...
                rShared);

            fig = figure( ...
                'Color', 'w', ...
                'Visible', 'off', ...
                'Units', 'pixels', ...
                'Position', [100 100 1200 800]);

            plot( ...
                tPlot, ...
                zBed, ...
                'LineWidth', 3);

            hold on;

            plot( ...
                tPlot, ...
                zMove, ...
                'LineWidth', 3);

            xline( ...
                0, ...
                'k-', ...
                'LineWidth', 1.8);


            xlabel( ...
                'Time relative to speech onset (s)');

            ylabel( ...
                'Shared projection');

            title( ...
                sprintf( ...
                'Shared PLSC dimension %d, r = %.3f', ...
                dimIdx, ...
                rShared));

            legend( ...
                {'Bed', 'Walking'}, ...
                'Location', 'best', ...
                'Box', 'off');

            ax = gca;

            ax.TickLength = [0 0];
            ax.FontSize = 20;
            ax.LineWidth = 1.5;
            ax.Box = 'off';

            grid off;

            xlim([tPlot(1), tPlot(end)]);

            figureBaseName = fullfile( ...
                sharedProjectionFolder, ...
                sprintf( ...
                '%s_shared_dim_%03d_projection', ...
                wordTag, ...
                dimIdx));

            exportgraphics( ...
                fig, ...
                [figureBaseName, '.png'], ...
                'Resolution', 300);

            savefig( ...
                fig, ...
                [figureBaseName, '.fig']);

            close(fig);

        end

    end


    %% =========================================================
    %  4. Save temporal projections of up to the first 10 unique dimensions
    %% =========================================================

    if isfield(resultStruct, 'unique') && ...
            isfield(resultStruct.unique, 'projectedData1') && ...
            isfield(resultStruct.unique, 'projectedData2') && ...
            ~isempty(resultStruct.unique.projectedData1) && ...
            ~isempty(resultStruct.unique.projectedData2)

        nUniquePlot = min([ ...
            10, ...
            numel(UniIdx), ...
            size(resultStruct.unique.projectedData1,2), ...
            size(resultStruct.unique.projectedData2,2)]);

        for iUnique = 1:nUniquePlot

            dimIdx = UniIdx(iUnique);

            zBed = ...
                resultStruct.unique.projectedData1(:,iUnique);

            zMove = ...
                resultStruct.unique.projectedData2(:,iUnique);

            if numel(t) == numel(zBed)

                tPlot = t;

            else

                tPlot = analysisStartTime + ...
                    (0:numel(zBed)-1) / fs_real;

            end

            rUnique = corr( ...
                zBed, ...
                zMove, ...
                'Rows', 'complete');

            fprintf( ...
                'Unique dimension %d corr = %.4f\n', ...
                dimIdx, ...
                rUnique);

            fig = figure( ...
                'Color', 'w', ...
                'Visible', 'off', ...
                'Units', 'pixels', ...
                'Position', [100 100 1200 800]);

            plot( ...
                tPlot, ...
                zBed, ...
                'LineWidth', 3);

            hold on;

            plot( ...
                tPlot, ...
                zMove, ...
                'LineWidth', 3);

            xline( ...
                0, ...
                'k-', ...
                'LineWidth', 1.8);


            xlabel( ...
                'Time relative to speech onset (s)');

            ylabel( ...
                'Unique projection');

            title( ...
                sprintf( ...
                'Unique dimension %d, r = %.3f', ...
                dimIdx, ...
                rUnique));

            legend( ...
                {'Bed', 'Walking'}, ...
                'Location', 'best', ...
                'Box', 'off');

            ax = gca;

            ax.TickLength = [0 0];
            ax.FontSize = 20;
            ax.LineWidth = 1.5;
            ax.Box = 'off';

            grid off;

            xlim([tPlot(1), tPlot(end)]);

            figureBaseName = fullfile( ...
                uniqueProjectionFolder, ...
                sprintf( ...
                '%s_unique_dim_%03d_projection', ...
                wordTag, ...
                dimIdx));

            exportgraphics( ...
                fig, ...
                [figureBaseName, '.png'], ...
                'Resolution', 300);

            savefig( ...
                fig, ...
                [figureBaseName, '.fig']);

            close(fig);

        end

    end


    %% =========================================================
    %  5. Prepare channel and frequency-band labels
    %% =========================================================

    sorted_labels = strings(0,1);

    Nch = numel(svd_channels);
    Nband = numel(svd_band);

    if Nband == 12

        bandLabels = [
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

        bandLabels = ...
            "Band " + string((1:Nband).');

    end


    %% Obtain anatomical labels for both contacts of each bipolar channel

    if all(svd_channels(:) + 1 <= height(channel_label_all))

        regionLabel1 = string( ...
            channel_label_all.StandardRegion(svd_channels));

        regionLabel2 = string( ...
            channel_label_all.StandardRegion(svd_channels + 1));

        bipolarLabels = ...
            regionLabel1 + " / " + regionLabel2;

    else

        regionLabel1 = strings(Nch,1);
        regionLabel2 = strings(Nch,1);

        bipolarLabels = ...
            "Channel " + string(svd_channels(:));

    end


    %% Reorder channels according to anatomical region labels

    [~, reorderChannel] = sort( ...
        lower(bipolarLabels));

    sorted_labels = ...
        bipolarLabels(reorderChannel);

    sortedChannelNumbers = ...
        svd_channels(reorderChannel);


    %% =========================================================
    %  6. Save loading heatmaps for significant shared dimensions
    %% =========================================================

    if numSigDims > 0 && ...
            isfield(resultStruct, 'shared') && ...
            isfield(resultStruct.shared, 'Loading1') && ...
            isfield(resultStruct.shared, 'Loading2') && ...
            ~isempty(resultStruct.shared.Loading1) && ...
            ~isempty(resultStruct.shared.Loading2)

        nLoadingPlot = min([ ...
            numSigDims, ...
            numel(SigIdx), ...
            size(resultStruct.shared.Loading1,2), ...
            size(resultStruct.shared.Loading2,2), ...
            5]);

        for iLoading = 1:nLoadingPlot

            dimIdx = SigIdx(iLoading);

            wBed = ...
                resultStruct.shared.Loading1(:,iLoading);

            wMove = ...
                resultStruct.shared.Loading2(:,iLoading);

            %% Normalize loading vectors

            normBed = norm(wBed);
            normMove = norm(wMove);

            if normBed > 0 && isfinite(normBed)
                wBed = wBed / normBed;
            end

            if normMove > 0 && isfinite(normMove)
                wMove = wMove / normMove;
            end

            assert( ...
                numel(wBed) == Nch*Nband, ...
                'Bed loading length mismatch.');

            assert( ...
                numel(wMove) == Nch*Nband, ...
                'Walking loading length mismatch.');

            %% Feature arrangement:
            % Frequency band is the outer index and channel is the inner index
            % After reshaping: [channel × band]

            WBedSigned = reshape( ...
                wBed, ...
                Nch, ...
                Nband);

            WMoveSigned = reshape( ...
                wMove, ...
                Nch, ...
                Nband);

            WBed = abs(WBedSigned);
            WMove = abs(WMoveSigned);

            %% Reorder channels by anatomical region

            WBedSorted = ...
                WBed(reorderChannel,:);

            WMoveSorted = ...
                WMove(reorderChannel,:);

            % Transpose to [band × channel] for visualization
            WBedPlot = ...
                WBedSorted.';

            WMovePlot = ...
                WMoveSorted.';

            % Use the same color scale for both behavioral contexts
            commonMax = max( ...
                [WBedPlot(:); WMovePlot(:)]);

            if ~isfinite(commonMax) || commonMax <= 0
                commonMax = 1;
            end


            %% -------------------------------------------------
            % 6.1 Bed-condition loading
            %% -------------------------------------------------

            fig = figure( ...
                'Color', 'w', ...
                'Visible', 'off', ...
                'Units', 'pixels', ...
                'Position', [100 100 1500 600]);

            imagesc(WBedPlot);

            clim([0, commonMax]);

            cb = colorbar;
            cb.Label.String = ...
                'Normalized absolute loading';

            xlabel('Bipolar channel');
            ylabel('Frequency band (Hz)');

            title( ...
                sprintf( ...
                'Shared dimension %d loading: bed', ...
                dimIdx));

            ax = gca;

            ax.XTick = 1:Nch;
            ax.XTickLabel = string(sortedChannelNumbers);
            ax.XTickLabelRotation = 90;

            ax.YTick = 1:Nband;
            ax.YTickLabel = bandLabels;

            ax.TickLength = [0 0];
            ax.FontSize = 14;
            ax.LineWidth = 1.3;
            ax.Box = 'off';

            figureBaseName = fullfile( ...
                loadingFolder, ...
                sprintf( ...
                '%s_shared_dim_%03d_loading_bed', ...
                wordTag, ...
                dimIdx));

            exportgraphics( ...
                fig, ...
                [figureBaseName, '.png'], ...
                'Resolution', 300);

            savefig( ...
                fig, ...
                [figureBaseName, '.fig']);

            close(fig);


            %% -------------------------------------------------
            % 6.2 Walking-condition loading
            %% -------------------------------------------------

            fig = figure( ...
                'Color', 'w', ...
                'Visible', 'off', ...
                'Units', 'pixels', ...
                'Position', [100 100 1500 600]);

            imagesc(WMovePlot);

            clim([0, commonMax]);

            cb = colorbar;
            cb.Label.String = ...
                'Normalized absolute loading';

            xlabel('Bipolar channel');
            ylabel('Frequency band (Hz)');

            title( ...
                sprintf( ...
                'Shared dimension %d loading: walking', ...
                dimIdx));

            ax = gca;

            ax.XTick = 1:Nch;
            ax.XTickLabel = string(sortedChannelNumbers);
            ax.XTickLabelRotation = 90;

            ax.YTick = 1:Nband;
            ax.YTickLabel = bandLabels;

            ax.TickLength = [0 0];
            ax.FontSize = 14;
            ax.LineWidth = 1.3;
            ax.Box = 'off';

            figureBaseName = fullfile( ...
                loadingFolder, ...
                sprintf( ...
                '%s_shared_dim_%03d_loading_walking', ...
                wordTag, ...
                dimIdx));

            exportgraphics( ...
                fig, ...
                [figureBaseName, '.png'], ...
                'Resolution', 300);

            savefig( ...
                fig, ...
                [figureBaseName, '.fig']);

            close(fig);


            %% -------------------------------------------------
            % 6.3 Joint contribution across bed and walking contexts
            %% -------------------------------------------------

            % Use the product of absolute loadings as the joint-contribution measure:
            % The value is large when the feature has a large loading in both contexts
            jointContribution = ...
                WBedSorted .* WMoveSorted;

            jointContributionPlot = ...
                jointContribution.';

            jointMax = max( ...
                jointContributionPlot(:));

            if ~isfinite(jointMax) || jointMax <= 0
                jointMax = 1;
            end

            fig = figure( ...
                'Color', 'w', ...
                'Visible', 'off', ...
                'Units', 'pixels', ...
                'Position', [100 100 1500 600]);

            imagesc(jointContributionPlot);

            clim([0, jointMax]);

            cb = colorbar;
            cb.Label.String = ...
                'Joint absolute loading product';

            xlabel('Bipolar channel');
            ylabel('Frequency band (Hz)');

            title( ...
                sprintf( ...
                'Shared dimension %d joint cross-context contribution', ...
                dimIdx));

            ax = gca;

            ax.XTick = 1:Nch;
            ax.XTickLabel = string(sortedChannelNumbers);
            ax.XTickLabelRotation = 90;

            ax.YTick = 1:Nband;
            ax.YTickLabel = bandLabels;

            ax.TickLength = [0 0];
            ax.FontSize = 14;
            ax.LineWidth = 1.3;
            ax.Box = 'off';

            figureBaseName = fullfile( ...
                loadingFolder, ...
                sprintf( ...
                '%s_shared_dim_%03d_loading_joint', ...
                wordTag, ...
                dimIdx));

            exportgraphics( ...
                fig, ...
                [figureBaseName, '.png'], ...
                'Resolution', 300);

            savefig( ...
                fig, ...
                [figureBaseName, '.fig']);

            close(fig);


            %% =================================================
            % 7. Save the top 10 channel-frequency features by joint contribution
            %% =================================================

            jointFeatureValue = ...
                abs(wBed) .* abs(wMove);

            [~, sortedFeatureIdx] = sort( ...
                jointFeatureValue, ...
                'descend');

            topK = min( ...
                10, ...
                numel(sortedFeatureIdx));

            idxTop = ...
                sortedFeatureIdx(1:topK);

            % Original feature order: frequency band as the outer index and channel as the inner index
            bandIndex = ...
                ceil(idxTop / Nch);

            channelPosition = ...
                1 + mod(idxTop-1, Nch);

            actualChannelIndex = ...
                svd_channels(channelPosition);

            topRegion1 = ...
                regionLabel1(channelPosition);

            topRegion2 = ...
                regionLabel2(channelPosition);

            topBandLabel = ...
                bandLabels(bandIndex);

            topFeatureTable = table( ...
                (1:topK).', ...
                idxTop(:), ...
                actualChannelIndex(:), ...
                bandIndex(:), ...
                topBandLabel(:), ...
                topRegion1(:), ...
                topRegion2(:), ...
                wBed(idxTop(:)), ...
                wMove(idxTop(:)), ...
                jointFeatureValue(idxTop(:)), ...
                'VariableNames', { ...
                'Rank', ...
                'FeatureIndex', ...
                'BipolarChannelIndex', ...
                'BandIndex', ...
                'BandHz', ...
                'Region1', ...
                'Region2', ...
                'BedLoading', ...
                'WalkingLoading', ...
                'JointLoadingProduct'});

            disp(topFeatureTable);

            topFeatureFile = fullfile( ...
                topFeatureFolder, ...
                sprintf( ...
                '%s_shared_dim_%03d_top_features.csv', ...
                wordTag, ...
                dimIdx));

            writetable( ...
                topFeatureTable, ...
                topFeatureFile);

        end

    end




    %% Organize outputs into a standardized analysis structure
    PLSCAnalysis = struct;

    PLSCAnalysis.schemaVersion = "PLSC_analysis_v2.1";
    PLSCAnalysis.createdAt = datetime( ...
        'now', 'Format', 'yyyy-MM-dd HH:mm:ss');

    PLSCAnalysis.subject = string(which_patient);
    PLSCAnalysis.wordIndex = diff_word;
    PLSCAnalysis.wordTag = string(wordTag);

    PLSCAnalysis.parameters = struct;
    PLSCAnalysis.parameters.fsInput = fs_re;
    PLSCAnalysis.parameters.fsOutput = fs_real;
    PLSCAnalysis.parameters.smoothMs = smooth_ms;
    PLSCAnalysis.parameters.analysisWindowSec = [bed_start_time - 9, bed_end_time - 9];
    PLSCAnalysis.parameters.minLagSec = minLagSec;
    PLSCAnalysis.parameters.minLagPts = minLagPts;
    PLSCAnalysis.parameters.significanceQuantile = sigThr;
    PLSCAnalysis.parameters.upperTailAlpha = 1-sigThr;
    PLSCAnalysis.parameters.standardization = ...
        "Feature-wise z-score separately within bed and move conditions";
    PLSCAnalysis.parameters.significanceRule = ...
        "Both covariance and correlation must exceed the null quantile; only consecutive significant dimensions from dimension 1 are retained";

    PLSCAnalysis.input = struct;
    PLSCAnalysis.input.dataBed = data_bed;
    PLSCAnalysis.input.dataMove = data_move;
    PLSCAnalysis.input.dataBedZ = data_bed_z;
    PLSCAnalysis.input.dataMoveZ = data_move_z;
    PLSCAnalysis.input.nTimePoints = size(data_bed_z,1);
    PLSCAnalysis.input.nFeatures = size(data_bed_z,2);

    PLSCAnalysis.features = struct;
    PLSCAnalysis.features.svdChannels = svd_channels(:);
    PLSCAnalysis.features.svdBands = svd_band(:);
    PLSCAnalysis.features.nChannels = numel(svd_channels);
    PLSCAnalysis.features.nBands = numel(svd_band);

    if exist('channel_label_all','var') && ...
            all(svd_channels(:)+1 <= height(channel_label_all))

        PLSCAnalysis.features.region1 = ...
            string(channel_label_all.StandardRegion(svd_channels));

        PLSCAnalysis.features.region2 = ...
            string(channel_label_all.StandardRegion(svd_channels+1));
    else
        PLSCAnalysis.features.region1 = strings(0,1);
        PLSCAnalysis.features.region2 = strings(0,1);
    end

    PLSCAnalysis.significance = nullStats;
    PLSCAnalysis.significance.covDist = covDist;
    PLSCAnalysis.significance.corrDist = corrDist;

    PLSCAnalysis.significance.validShiftsSec = ...
        nullStats.validShifts / fs_real;


    PLSCAnalysis.dimensions = struct;
    PLSCAnalysis.dimensions.numSignificant = numSigDims;
    PLSCAnalysis.dimensions.significantIndices = SigIdx(:);
    PLSCAnalysis.dimensions.pointwiseNonSignificantIndices = ...
        PointwiseNonSigIdx(:);
    PLSCAnalysis.dimensions.nonRetainedIndices = ...
        NonRetainedIdx(:);
    PLSCAnalysis.dimensions.plottedNonSignificantIndices = ...
        NonSigPlotIdx(:);
    PLSCAnalysis.dimensions.uniqueIndices = UniIdx(:);

    PLSCAnalysis.neuralSpace = resultStruct;

    if exist('sorted_labels','var')
        PLSCAnalysis.loadingLabels = sorted_labels;
    else
        PLSCAnalysis.loadingLabels = strings(0,1);
    end

    PLSCAnalysis.paths = struct;
    PLSCAnalysis.paths.wordFolder = string(wordSaveFolder);
    PLSCAnalysis.paths.figureFolder = string(figureFolder);
    PLSCAnalysis.paths.dataFolder = string(dataFolder);

    % Save the complete analysis structure in a single MAT file
    analysisMatFile = fullfile( ...
        dataFolder, ...
        sprintf('%s_PLSC_analysis.mat', wordTag));

    save(analysisMatFile, 'PLSCAnalysis', '-v7.3');

    % Save the complete statistics table for all dimensions
    summaryCsvFile = fullfile( ...
        dataFolder, ...
        sprintf('%s_PLSC_dimension_statistics.csv', wordTag));

    writetable(nullStats.resultTable, summaryCsvFile);

    % Save the analysis parameters
    parameterName = [
        "subject"
        "wordIndex"
        "fsInput"
        "fsOutput"
        "smoothMs"
        "analysisStartSec"
        "analysisEndSec"
        "minLagSec"
        "minLagPts"
        "significanceQuantile"
        "nTimePoints"
        "nFeatures"
        "nChannels"
        "nBands"
        "numSignificantDimensions"
        "numUniqueShifts"
        ];

    parameterValue = [
        string(which_patient)
        string(diff_word)
        string(fs_re)
        string(fs_real)
        string(smooth_ms)
        string(bed_start_time - 9)
        string(bed_end_time - 9)
        string(minLagSec)
        string(minLagPts)
        string(sigThr)
        string(size(data_bed_z,1))
        string(size(data_bed_z,2))
        string(numel(svd_channels))
        string(numel(svd_band))
        string(numSigDims)
        string(nullStats.numShifts)
        ];

    parameterTable = table( ...
        parameterName, parameterValue, ...
        'VariableNames', {'Parameter','Value'});

    parameterCsvFile = fullfile( ...
        dataFolder, ...
        sprintf('%s_PLSC_parameters.csv', wordTag));

    writetable(parameterTable, parameterCsvFile);

end

end

%% Local function: save covariance and correlation null distributions for one dimension
function savePLSCNullFigure( ...
    dimIdx, covDist, corrDist, nullStats, ...
    figureBaseName, dimensionStatus)

nullCov = covDist(dimIdx,:);
nullCorr = corrDist(dimIdx,:);

nullCov = nullCov(isfinite(nullCov));
nullCorr = nullCorr(isfinite(nullCorr));

obsCov = nullStats.obsCov(dimIdx);
thrCov = nullStats.thrCov(dimIdx);
pCov = nullStats.pCov(dimIdx);

obsCorr = nullStats.obsCorr(dimIdx);
thrCorr = nullStats.thrCorr(dimIdx);
pCorr = nullStats.pCorr(dimIdx);

nBinsCov = min(30, max(10, round(sqrt(numel(nullCov)))));
nBinsCorr = min(30, max(10, round(sqrt(numel(nullCorr)))));

fig = figure( ...
    'Color','w', ...
    'Visible','off', ...
    'Units','pixels', ...
    'Position',[100 100 1300 520]);

tl = tiledlayout(1,2, ...
    'TileSpacing','compact', ...
    'Padding','compact');

nexttile;

histogram(nullCov, nBinsCov, ...
    'Normalization','probability');
hold on;

xline(obsCov,'r-', ...
    'LineWidth',2.5, ...
    'Label','Observed');

xline(thrCov,'k--', ...
    'LineWidth',2.5, ...
    'Label',sprintf('%.1f%% threshold', ...
    nullStats.sigThr*100));

xlabel('Cross-context covariance');
ylabel('Probability');

title(sprintf( ...
    'Covariance: obs=%.4g, threshold=%.4g, p=%.4g', ...
    obsCov, thrCov, pCov));

ax = gca;
ax.TickLength = [0 0];
ax.FontSize = 16;
ax.LineWidth = 1.5;
box off;
grid off;

nexttile;

histogram(nullCorr, nBinsCorr, ...
    'Normalization','probability');
hold on;

xline(obsCorr,'r-', ...
    'LineWidth',2.5, ...
    'Label','Observed');

xline(thrCorr,'k--', ...
    'LineWidth',2.5, ...
    'Label',sprintf('%.1f%% threshold', ...
    nullStats.sigThr*100));

xlabel('Projection correlation');
ylabel('Probability');

title(sprintf( ...
    'Correlation: obs=%.4f, threshold=%.4f, p=%.4g', ...
    obsCorr, thrCorr, pCorr));

ax = gca;
ax.TickLength = [0 0];
ax.FontSize = 16;
ax.LineWidth = 1.5;
box off;
grid off;

title(tl, sprintf( ...
    '%s PLSC dimension %d | CovSig=%d, CorrSig=%d', ...
    dimensionStatus, dimIdx, ...
    nullStats.dimCov(dimIdx), ...
    nullStats.dimCorr(dimIdx)), ...
    'FontSize',18, ...
    'FontWeight','bold');

exportgraphics( ...
    fig, figureBaseName + ".png", ...
    'Resolution',300);

savefig(fig, figureBaseName + ".fig");

close(fig);

end





%% Additional helper functions

function [resultStruct_word1,resultStruct_word2,resultStruct_word3,resultStruct_word4] = loadresultStruct(which_patient)

tmp1 = load(["G:/wxl/"+which_patient+"/results/data/realneed/resultStruct_word1.mat"]);
tmp2 = load(["G:/wxl/"+which_patient+"/results/data/realneed/resultStruct_word2.mat"]);
tmp3 = load(["G:/wxl/"+which_patient+"/results/data/realneed/resultStruct_word3.mat"]);
tmp4 = load(["G:/wxl/"+which_patient+"/results/data/realneed/resultStruct_word4.mat"]);

resultStruct_word1 = tmp1.resultStruct;
resultStruct_word2 = tmp2.resultStruct;
resultStruct_word3 = tmp3.resultStruct;
resultStruct_word4 = tmp4.resultStruct;

end


%%

function [channel_label_all] = getChannellabel(which_patient,patient_idx)
% Path to the channel-label file

LoadPath = ["G:/wxl/"+which_patient+"/"+patient_idx+"电极通道对应关系.xlsx"];
channel_label_all = readtable(LoadPath);
channel_label_all.Properties.VariableNames = {'Var1','Var2','Var3','Var4','Var5','Var6','Var7'}; % Rename table variables to standardized names
channel_label = channel_label_all(:,[3]);
channel_label=table2array(channel_label);

end


%% PLSC null distribution based on circular time shifts
function [numSigDims, covDist, corrDist, SigIdx, nullStats] = ...
    computeSharedNullDistribution(data1, data2, minLagPts, sigThr)

if nargin < 3 || isempty(minLagPts)
    minLagPts = 30;
end

if nargin < 4 || isempty(sigThr)
    sigThr = 0.975;
end

if size(data1,1) ~= size(data2,1)
    error('data1和data2的时间点数必须相同。');
end

if size(data1,2) ~= size(data2,2)
    error('data1和data2的特征数必须相同。');
end

if any(~isfinite(data1(:))) || any(~isfinite(data2(:)))
    error('PLSC输入包含NaN或Inf。');
end

nTime = size(data1,1);

if nTime <= 2*minLagPts
    error('必须满足minLagPts<nTime/2。');
end

[~,~,singularValues,projectedData1,projectedData2] = ...
    plsc(data1,data2);

obsCov = diag(singularValues);

obsCorr = diag(corr( ...
    projectedData1, projectedData2, ...
    'Rows','complete'));

nDims = min(numel(obsCov),numel(obsCorr));

obsCov = obsCov(1:nDims);
obsCorr = obsCorr(1:nDims);

validShifts = minLagPts:(nTime-minLagPts);
numSims = numel(validShifts);

fprintf(['Null distribution: %d unique circular shifts, ', ...
    'shift range %d-%d samples.\n'], ...
    numSims, validShifts(1), validShifts(end));

permCov = nan(nDims,numSims);
permCorr = nan(nDims,numSims);

parfor sim = 1:numSims

    data2Perm = circshift( ...
        data2, validShifts(sim), 1);

    [~,~,singularValuesPerm, ...
        projectedPerm1,projectedPerm2] = ...
        plsc(data1,data2Perm);

    covPerm = diag(singularValuesPerm);

    corrPerm = diag(corr( ...
        projectedPerm1,projectedPerm2, ...
        'Rows','complete'));

    permCov(:,sim) = covPerm(1:nDims);
    permCorr(:,sim) = corrPerm(1:nDims);

end

thrCov = prctile(permCov,sigThr*100,2);
thrCorr = prctile(permCorr,sigThr*100,2);

dimCov = obsCov > thrCov;
dimCorr = obsCorr > thrCorr;
dimAll = dimCov & dimCorr;

pCov = ...
    (sum(permCov >= obsCov,2)+1) ...
    /(numSims+1);

pCorr = ...
    (sum(permCorr >= obsCorr,2)+1) ...
    /(numSims+1);

covMargin = obsCov-thrCov;
corrMargin = obsCorr-thrCorr;

firstNonSig = find(~dimAll,1,'first');

if isempty(firstNonSig)
    SigIdx = (1:nDims).';
else
    SigIdx = (1:firstNonSig-1).';
end

numSigDims = numel(SigIdx);

isSequentiallyRetained = ...
    ismember((1:nDims).',SigIdx);

resultTable = table( ...
    (1:nDims).', ...
    obsCov,thrCov,covMargin,pCov, ...
    obsCorr,thrCorr,corrMargin,pCorr, ...
    dimCov,dimCorr,dimAll, ...
    isSequentiallyRetained, ...
    'VariableNames',{ ...
    'Dimension', ...
    'ObservedCov', ...
    'CovThreshold', ...
    'CovMargin', ...
    'CovP', ...
    'ObservedCorr', ...
    'CorrThreshold', ...
    'CorrMargin', ...
    'CorrP', ...
    'CovSignificant', ...
    'CorrSignificant', ...
    'BothSignificant', ...
    'SequentiallyRetained'});

disp(resultTable);

covDist = permCov;
corrDist = permCorr;

nullStats = struct;

nullStats.obsCov = obsCov;
nullStats.obsCorr = obsCorr;

nullStats.thrCov = thrCov;
nullStats.thrCorr = thrCorr;

nullStats.covMargin = covMargin;
nullStats.corrMargin = corrMargin;

nullStats.pCov = pCov;
nullStats.pCorr = pCorr;

nullStats.dimCov = dimCov;
nullStats.dimCorr = dimCorr;
nullStats.dimAll = dimAll;

nullStats.SigIdx = SigIdx;
nullStats.numSigDims = numSigDims;

nullStats.validShifts = validShifts(:);
nullStats.numShifts = numSims;

nullStats.minLagPts = minLagPts;
nullStats.sigThr = sigThr;

nullStats.resultTable = resultTable;

end

