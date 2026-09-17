%% time bin analysis:

% create time bin structures from the callsTable in which each time bin is
% about 40 minutes. we will call them early, middle and late. take amplitude
% and call counts from the batalef table.Then do the same statical
% analysis. 

%% create time bins:
% *********load the CallsTble from batalef*********

% open batalef and import data then;
% numCalls = max(callsTable.CallIdx); % total number of calls 
timeMat = sortrows([callsTable.CallPeakTime , callsTable.CallPeakPower,...
                             callsTable.CallIdx , callsTable.ChannelIdx]); % [ detecion time , peak , call indx , geoph indx ] 

% Remove entries with amplitude close to 0:
validAmp = timeMat(:,2) < -10; % valid data mask
beamData = timeMat(validAmp,:);

% create 3 time bins:
% timeBins = linspace(min(beamData(:,1)), max(beamData(:,1)), 4);
timeBins = linspace(0,7200,4);
binStruct_R5S2 = cell(3,1); % to store data for each individual session 

for numBins = 1:3

    bin_values = (beamData(:,1)>=timeBins(numBins)) & (beamData(:,1)<timeBins(numBins+1)); % values within the valid bin mask
    
    callPower = beamData(bin_values,2); % amplitudes within the bin
    callTime = beamData(bin_values,1); % time point within the bin
    numCalls = numel(unique(beamData(bin_values,3))); % total number of calls 
    %--------------------------------------------------------------------------------------------------------------------------------
   
    % Store data for plotting and analysis:
    binStruct_R5S2{numBins} = struct('time', callTime,'powers', callPower,'calls', numCalls);

end

%---------------------------------------------------------------------------------
%% amplitudes shown in box plots:
% relevent data in: X:\Users\Members\Ragad\Grace BMR project\results\amplitude_structures
clear; clc; close all;
load("binStruct_R6S1.mat");

earlyBin = binStruct_R5S2{1}.powers;
middleBin =binStruct_R5S2{2}.powers;
lateBin = binStruct_R5S2{3}.powers;

avgAmp = mean([ earlyBin  ; middleBin  ; lateBin],"all");
%------------------
earlyTime = binStruct_R5S2{1}.time;
middleTime = binStruct_R5S2{2}.time;
lateTime = binStruct_R5S2{3}.time;
%------------------
if isempty(middleBin)
    middleBin = 1;
    middleTime=1;

end

if isempty(earlyBin)
    earlyBin = 1;
    earlyTime = 1;
end

if isempty(lateBin)
    lateBin = 1;
    lateTime = 1;
end
%------------------
powerData = [earlyBin ; middleBin ; lateBin];
g = [ones(size(earlyBin)); 2*ones(size(middleBin)); 3*ones(size(lateBin))];

figure()
boxplot(powerData,g, "Labels",{'Early', 'Middle','Late'})
ylabel("Amplitude (db)"); title('R1S1: Temporal Change of Beam Pattern');

%--------------------------------------------------------------------------------
%% statistics:
fprintf('1. DESCRIPTIVE STATISTICS:\n');
fprintf('%-10s %-10s %-16s %-8s %-8s %-8s %-8s\n', 'Bin', 'Mean', 'Latency', 'Calls', 'Min', 'Max', 'N');
fprintf('%-10s %-10s %-16s %-8s %-8s %-8s %-8s\n', '---', '----', '---------------', '-----', '---', '---', '-');

bins_data = {earlyBin, middleBin, lateBin};
bin_time = {earlyTime, middleTime, lateTime};

descriptive_stats = zeros(3, 6); % mean, median, std, min, max, n
binNames = cell(1,3);

% for numCoparison = 1:3
%     data = bins_data{numCoparison};
%     descriptive_stats(numCoparison,:) = [mean(data), median(data), std(data), min(data), max(data), length(data)];
%     fprintf('%-10s %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f %-8.0f\n', ...
%         binNames{numCoparison}, descriptive_stats(numCoparison,:));
% end

for numCoparison = 1:3
    data       = bins_data{numCoparison};
    firstTime  = bin_time{numCoparison}(1);
    calls      = binStruct_R5S2{numCoparison}.calls;

    descriptive_stats(numCoparison,:) = [mean(data), firstTime, calls, min(data), max(data), length(data)];

    fprintf('%-10s %-10.2f %-16.3f %-8d %-8.2f %-8.2f %-8.0f\n', ...
        binNames{numCoparison}, mean(data), firstTime, calls, min(data), max(data), length(data));
end
fprintf('\n');

%--------------------------------------------------------------------------