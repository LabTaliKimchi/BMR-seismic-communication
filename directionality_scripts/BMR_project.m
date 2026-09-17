%% BMR_project
% this code was used to calculate the the speed of propagation and the
% attenuation of the seismic signals
%% Variables:
% open batalef and import data then;
numCalls = max(callsTable.CallIdx);
timeMat = sortrows([callsTable.CallIdx,callsTable.ChannelIdx,callsTable.CallPeakTime]);
peaksMat = sortrows([callsTable.CallIdx,callsTable.ChannelIdx,callsTable.CallPeakEnvValue]);


XY = mics_cart_positions(:,1:2);
EucDist = sqrt(sum(XY.^2, 2));


% %--------------------------------------------------------------------------
% %Save peaksMat & EucDist for python processing(krining)
% % ### Already used and altered in matlab. please do not run these line
% % without changing the paths. 
%filepathEuc = 'D:\Users\Administrator\Desktop\George\Projects\BMR_Geo\Python analysis\Data\R6S1\XYpy.csv' ;
% %writematrix(XY,filepathEuc) ;
filepathpeak = 'D:\Users\Administrator\Desktop\George\Projects\BMR_Geo\Python analysis\Data\R3S1\R31_S1_peaksMatpy.csv' ;
writematrix(peaksMat,filepathpeak);
% % %
filepathtime = 'D:\Users\Administrator\Desktop\George\Projects\BMR_Geo\Python analysis\Data\R3S1\R3_S1_timeMatpy.csv' ;
writematrix(timeMat,filepathtime)
% 
%  %--------------------------------------------------------------------------
% % %% Findng the lowdest geophpnes:
% % 
% % t = [callsTable.CallIdx,callsTable.ChannelIdx,callsTable.CallPeakTime];
% % p = [callsTable.CallIdx,callsTable.ChannelIdx,callsTable.CallPeakPower];
% % ff = 'C:\Users\Administrator\Desktop\figures\mics'; % change folder path
% % 
% % for micIndx = 1:50
% %     figure()
% %     set(gcf,'position',[39,408,1833,420])
% %     plot(t(t(:,2)==micIndx,3), p(p(:,2)==micIndx,3),Marker="*",LineStyle="--")
% %     xlabel('Time (sec)',FontSize=12); ylabel('Peak amplitude (db)',FontSize=12); 
% %     title(['Mic - ', num2str(micIndx)],"FontSize",17); grid; xlim([1900 2600])
% % 
% %     % Save the figure as an image in the folder
% %     filename = sprintf('Mic_%d.jpg',micIndx); % Generate a filename
% %     filepath = fullfile(ff, filename); % Create the full filepath
% %     saveas(gcf, filepath); % Save the figure
% %     close;
% % end
% %--------------------------------------------------------------------------
%% BEAM DECAY (LOOKING AT INDIVIDUAL CALLS):
fitting_mat_decay = cell(numCalls,1);
decayCoeffs = cell(numCalls,1);

countr = 0;

%folder = 'E:\george_correction\beam_deacy_single_calls'; % change folder path
folder = 'D:\Users\Administrator\Desktop\George\Projects\BMR_Geo\Matlab Figures\Beam_Decay\R3S1\Singles' ; %Folder path :George PC
for callIndx = 1:numCalls

    % BEAM DECAY:
    beamPeak = sortrows(peaksMat(peaksMat(:,1)==callIndx,:),3); % a 3x3 matrix --> [call_index ,  mic_index , peak_power] sorted based on power
    channelOrderPeaks = beamPeak(:,2);
    distCoordinatesPeaks = XY(channelOrderPeaks,:); % the euclidian distance from the center point (0,0)
    distancePeaks = sqrt(sum(distCoordinatesPeaks.^2, 2)); % the euclidian distance from the center point (0,0)
    %------------------------------
    % GET RID OF NAN:
    beamsVec = ~isnan(beamPeak(:,end));
    %------------------------------
    % fitting curve:
    fitPlot = fit(distancePeaks(beamsVec),beamPeak(beamsVec,end),"poly1");
    coefVal = coeffvalues(fitPlot); % fit = a*X + c
    if coefVal(1)>0
           countr = countr+1;
    end

    % distCoordinatesPeaks = distMat(channelOrderPeaks,:);
    % distancePeaks = diag(pdist2(distCoordinatesPeaks(:,1),distCoordinatesPeaks(:,2),"euclidean")); % take main diagonal
    % SAVE THE DATA FOR FITTIGN:
    % fitting_mat_decay{callIndx} = [distancePeaks beamPeak(:,end)]; % FOR SECTION: BEAM DECAY (ALL CALLS) 

    fitEquation = [num2str(coefVal(1),'%.2f'),'*x + ',num2str(coefVal(2),'%.2f')];
    decayCoeffs{callIndx} = coefVal(1);
    %----------------------------------------------------------------------
    figure()
    set(gcf,'position',[555,409,1169,386]) 

    subplot(1,2,1)
    plot(fitPlot,distancePeaks(beamsVec),beamPeak(beamsVec,end),'*')
    grid; xlabel("Distance (cm)","FontSize",12); ylabel("peak amplitude (db)","FontSize",12);
    title(['Beam decay = ',fitEquation],"FontSize",15); %ylim([0 0.6]); xlim([0 140]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % AMPLITUDE AND MICS:
    subplot(1,2,2)
    plot(beamPeak(:,2),beamPeak(:,end),Marker="o",LineStyle="none",MarkerFaceColor=[0 0.4470 0.7410])
    grid; xlabel("Mic number","FontSize",12); ylabel("peak amplitude (db)","FontSize",12);
    title("Amplitude to mic number","FontSize",15);
    %----------------------------------------------------------------------
    sgtitle(['R3S1 Call number ', num2str(callIndx)],"FontSize",17); % change this so that it changes for each call
    % Save the figure as an image in the folder
    filename = sprintf('call_%d.jpg',callIndx); % Generate a filename
    filepath = fullfile(folder, filename); % Create the full filepath
    % saveas(gcf, filepath); % Save the figure
    close;
end

%--------------------------------------------------------------------------
%% BEAM DECAY (ALL CALLS):

% FITTING CURVE FOR AMPLITUDE ATTINUATION:
decayData = cell2mat(fitting_mat_decay);
decayDataVec = ~isnan(decayData(:,2)); % to remove NaNs
distances = decayData(decayDataVec,1); 
amplitudes = decayData(decayDataVec,2);

fitPlot2 = fit(distances,amplitudes,"poly1");
coefDecay = coeffvalues(fitPlot2); 
fitEquation = [num2str(coefDecay(1),'%.2f'),'*X + ',num2str(coefDecay(2),'%.2f')];
%--------------------------------------------------------------------------
figure()
plot(fitPlot2,distances,amplitudes)
xlabel("Distance (cm)","FontSize",15); ylabel("peak amplitude (db)","FontSize",15);
title(['Beam Decay = ' , fitEquation],'FontSize',15); xlim([0 220]); grid; 

%--------------------------------------------------------------------------
% Calculate mean amplitude for each unique distance
uniqueDistances = unique(distances);
meanAmplitudes = zeros(length(uniqueDistances), 1);

for ii = 1:length(uniqueDistances)
    idnx = distances == uniqueDistances(ii);
    meanAmplitudes(ii) = mean(amplitudes(idnx));
end

figure()
plot(fitPlot2,uniqueDistances,meanAmplitudes)
xlabel("Distance (cm)","FontSize",15); ylabel("peak amplitude (db)","FontSize",15);
title(['Beam Decay = ' , fitEquation],'FontSize',15); xlim([0 220]); grid; 
f = gcf;

save("average_amps_R6S1.mat","meanAmplitudes");
% save("distances_R4S1.mat","uniqueDistances");
%--------------------------------------------------------------------------

% PLOT COEFFS:
figure()
histogram(cell2mat(decayCoeffs), 'Normalization', 'probability');
xline(coefDecay(1),Color='r',LineWidth=2); xlim([-0.2 0])
xlabel('decay values',FontSize=15); ylabel('Probability',FontSize=15);
title({'Distribution of amplitude decay values',['Attenuation = ', num2str(coefDecay(1),'%.2f'),' db/cm'],...
    'Attenuation variance = ', num2str(var(cell2mat(decayCoeffs)))},fontsize=15);
%--------------------------------------------------------------------------
%% speed calc.
%BeamSpeed Segmented by Euc Dist 
%This segment is to prep the data to plot 
% figure();
% h=histogram(EucDist,11); % check histogram for appropriate geophone
%segmentation
MicNumb = (1:size(mics_cart_positions, 1))'; % other than R5S2 AND R4S2
% MicNumb = (1:49)'; % Currently working on R4S2
PreBin = [EucDist(:,1),MicNumb(:,1)];
% PreBin = [EucDist{:,1}, MicNumb];

Edges = [10 29 49.5 67 86 105 124 143 162 181 200 219]; % edges where extracted from a EucDist histogram 
%% speed calc.   
%The following loop will take out all Euc distances for each bin and their
%respective mic number, and save them in cell form 'rangevectors (1xn)'
for ii = 1:length(Edges)-1
    rangeMask = PreBin(:,1) >= Edges(ii) & PreBin(:,1) < Edges(ii+1);
    rangeVectors{ii} = PreBin(rangeMask, :);  % Keep both distance and mic #
end
 
% rangeVector CleanUp - remove far away/singualer 
singleEntryCount = 0;
filteredVectors = {};  

for ii = 1:length(rangeVectors)
    entry = rangeVectors{ii};

    if isnumeric(entry) && size(entry, 1) == 1
        singleEntryCount = singleEntryCount + 1;
        continue;  
    end
    filteredVectors{end+1} = entry;
end
%remove bins that have one microphone (12&50).
rangeVectors = filteredVectors;

%Activate to remove for any defective mics (Nans) Remove mics ; i.e R3S1
%Mic#36
%Try to find in which bin in "RangedVector" the defective mic is in"
matrix = rangeVectors{5};
rows_to_keep = matrix(:,2) ~= 36;
rangeVectors{5} = matrix(rows_to_keep, :);
% Display result
fprintf('Removed %d single-entry vectors. Remaining: %d\n', ...
    singleEntryCount, numel(rangeVectors));

%% speed calc.
% Initialize as empty numeric vector
averageDistances = [];        % empty vector
geoLabels = strings(0);       % empty string vector
for ii = 1:length(rangeVectors) % loops over our bin vector, checks if naan, if not -> calculates Average distance.
    thisEntry = rangeVectors{ii};  
    % Skip if cell is NaN or empty, leaves us with non-empty bins only.
    if isnumeric(thisEntry) && (all(isnan(thisEntry), 'all') || isempty(thisEntry))
        continue;
    end
    % Compute average of the first column (distance)
    avgDist = mean(thisEntry(:,1));
    
    % Store the result
    averageDistances(end+1) = avgDist;
    geoLabels(end+1) = 'Geo' + string(ii);
end
AvgDistGeo = [geoLabels',averageDistances']; 


callIDs = unique(timeMat(:,1));
numCalls = numel(callIDs);
numBins = length(rangeVectors);

% Creat empty result matrix: rows = n#calls, columns = n#bins
avgTimes = NaN(numCalls, numBins);  % NaN if no data for that bin


for idnx = 1:numCalls
    callID = callIDs(idnx);
    % Get all rows in callData that match this callID
    callRows = timeMat(timeMat(:,1) == callID, :);  % [MicNum, Time]
    
    for binIdx = 1:numBins
        micNumsInBin = rangeVectors{binIdx}(:,2);  % Mic numbers in current bin
        
        % Find matching mic rows for this call
        isMicInBin = ismember(callRows(:,2), micNumsInBin);
        timesInBin = callRows(isMicInBin, 3);  % Time column
        
        if ~isempty(timesInBin)
            avgTimes(callID, binIdx) = mean(timesInBin);
        end
    end
end
%remove all coloumn with Nan row i.e empty bins.
colsWithAllNaN = all(isnan(avgTimes), 1);
avgTimes(:, colsWithAllNaN) = []; 

%%Transform to deltaT - i.e T1 = 0 
avgTimes = avgTimes - avgTimes(:,1);

%--------------------------------------------------------------------------------
%% speed calc. -- Plots
%Plotting Single calls - > Figures saved to folder
% Convert distances from string to numeric
distances = str2double(AvgDistGeo(:,2));   % 9x1 numeric vector
%folder = 'D:\Users\Administrator\Desktop\George\Projects\BMR_Geo\Matlab Figures\Speed\R3S1\Singles'; % change folder path
folder = 'X:\Users\Members\Ragad\Grace BMR project\raw_data\R3S1\Speed singles'; % for Ragad 

% Loop over each call (row in avgTimes)
for ii = 1:size(avgTimes, 1)
    % Extract time values for this call
    times = avgTimes(ii, :);
    
    % Create the plot
    figure('Visible', 'off');  % Hide figure for speed
    plot(averageDistances, times, 'o-', 'LineWidth', 1.5);
    xlabel('Average Euclidean Distance');
    ylabel('Arrival Time');
    title(['Call #' num2str(ii)]);
    grid on;
    ylim([min(times,[],'omitnan') - 0.001, max(times,[],'omitnan') + 0.001]);

    % Save figure
    filename = fullfile(folder, ['R3S1 Call_' num2str(ii) '.png']);
    saveas(gcf, filename);
    close(gcf);  % Close the figure to save memory
end
%--------------------------------------------------------------------------
%% All on one plot

folder = 'X:\Users\Members\Ragad\Grace BMR project\raw_data\R3S1';
meanTimes = nanmean(avgTimes); 
stdTimes  = nanstd(avgTimes);    
upperBound = meanTimes + stdTimes;
lowerBound = meanTimes - stdTimes;
figure;
hold on ;
% Loop over each call and plot
for ii = 1:size(avgTimes, 1)
    times = avgTimes(ii, :);   
    % Skip empty rows
    if all(isnan(times))
        continue;
    end
    plot(distances, times, '-', 'Color', [0.7 0.7 0.7]); 
end

plot(distances, meanTimes, 'r-', 'LineWidth', 2.5);
plot(distances, upperBound, 'black', 'LineWidth', 1);
plot(distances, lowerBound, 'black', 'LineWidth', 1);
f_speed = fit(distances(:),meanTimes(:), 'poly1');
f_x = linspace(min(distances), max(distances), 100); % smoother fit line
f_y = f_speed(f_x);
plot(f_x, f_y,'Color', 'b' ,'LineStyle', ':', 'LineWidth', 2);  % blue regression line
xlabel('Bin Eucledean Distance (cm)');
ylabel('Arrival Time (sec)');
title('Call speed sec/cm');
grid on;
ylim([min(times,[],'omitnan') - 0.001, max(times,[],'omitnan') + 0.001]);
%legend('Individual Calls', 'Mean Arrival Time');
hold off;
% Save figure
filename = fullfile(folder, 'R3S1 All calls.png'); %% Choose the correct data file titl
saveas(gcf, filename);
%--------------------------------------------------------------------------
% %% Directional Beam amplitude - GeoMap - Excel file for python processing
% % Mics are alloctaed a direction based on map into 8 directional vectors.
% 
% calls = peaksMat(:,1);
% mics  = peaksMat(:,2);
% amps  = peaksMat(:,3);
% 
% % Normalize mic numbers to column indices
% [uniqueMics, ~, micIdx] = unique(mics);  % micIdx maps each mic# to 1:N
% numCalls = max(calls);
% numMics  = numel(uniqueMics);
% 
% % Convert call number to row index
% rowIdx = calls;
% 
% % Create matrix: rows = calls, columns = mics
% Kirging_Mat = accumarray([rowIdx, micIdx], amps, [numCalls, numMics], @mean, NaN);


