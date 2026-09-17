% load('A.mat');   % Uncomment if A is stored in A.mat
clear; clc; close all;

%% Load your matrix (if needed)
A = R6S1AllKrigedGrids;

% Check that A is big enough
if size(A,1) < 25 || size(A,2) < 25
    error('Matrix A must be at least 25x25.');
end

%% 1. Find the peak of the matrix
[peakVal, peakIdx] = max(A(:));
[rowPeak, colPeak] = ind2sub(size(A), peakIdx);

fprintf('Peak value: %g at (row=%d, col=%d)\n', peakVal, rowPeak, colPeak);

%% 2. Define the vector from (25,25) to the peak
startPoint = [61; 30];             % [row; col]
peakPoint  = [rowPeak; colPeak];   % [row; col]

v = peakPoint - startPoint;        % vector from (50,40) to the peak

% If the peak is exactly at (25,25), this will be zero-length
if norm(v) == 0
    error('Peak is at (25,25); direction vector is zero.');
end

v_unit = v / norm(v);              % unit vector in that direction

%% 3. Define a perpendicular vector (still in [row; col] space)
% A perpendicular vector to [a; b] is [-b; a]
v_perp = [-v_unit(2); v_unit(1)];  % also unit length

%% 4. Compute the longest possible cross-section through the peak,
%    perpendicular to v, staying inside the matrix

[nRows, nCols] = size(A);

% We parametrize the line as:
%   [row; col](t) = peakPoint + t * v_perp,  where t is scalar
% We want all points to satisfy:
%   1 <= row(t) <= nRows
%   1 <= col(t) <= nCols

t_min = -Inf;
t_max =  Inf;

% Row constraints
if v_perp(1) > 0
    t_min = max(t_min, (1     - peakPoint(1)) / v_perp(1));
    t_max = min(t_max, (nRows - peakPoint(1)) / v_perp(1));
elseif v_perp(1) < 0
    t_min = max(t_min, (nRows - peakPoint(1)) / v_perp(1));
    t_max = min(t_max, (1     - peakPoint(1)) / v_perp(1));
end

% Column constraints
if v_perp(2) > 0
    t_min = max(t_min, (1     - peakPoint(2)) / v_perp(2));
    t_max = min(t_max, (nCols - peakPoint(2)) / v_perp(2));
elseif v_perp(2) < 0
    t_min = max(t_min, (nCols - peakPoint(2)) / v_perp(2));
    t_max = min(t_max, (1     - peakPoint(2)) / v_perp(2));
end

% Sample points along this line
nSamples = 500; % resolution of the cross-section
t = linspace(t_min, t_max, nSamples);

rowLine = peakPoint(1) + t * v_perp(1);
colLine = peakPoint(2) + t * v_perp(2);

%% 5. Sample the matrix values along this perpendicular cross-section
% Use interpolation so we’re not restricted to integer indices
[colGrid, rowGrid] = meshgrid(1:nCols, 1:nRows); % note order: (x,y) = (col,row)

profileVals = interp2(colGrid, rowGrid, A, colLine, rowLine, 'linear');

%% 6. (Optional) Plot the matrix and the cross-section line

figure()

subplot(1,2,1)
imagesc(A);
axis image xy;
colormap parula;
colorbar;
hold on;

% Plot start point and peak
plot(startPoint(2), startPoint(1), 'wo', 'MarkerFaceColor', 'w', 'MarkerSize', 8);
plot(colPeak, rowPeak, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);

% Plot cross-section line
plot(colLine, rowLine, 'w-', 'LineWidth', 1.5);

title('Matrix with peak, start point, and perpendicular cross-section');
% legend({'Start (25,25)', 'Peak', 'Perpendicular cross-section'}, 'TextColor', 'w');
% --------------------------------------------------------------------------
% (Optional) Plot the cross-section profile
subplot(1,2,2)
plot(profileVals, 'LineWidth', 1.5);
xlabel('Sample index along perpendicular line');
ylabel('A value');
title('Cross-section through peak, perpendicular to vector from (25,25)');
grid on;

%% 8. Calculate FWHM (Full Width at Half Maximum) in degrees

% -->>>> I added this section <<<<--
% Find the maximum value in the profile
[maxVal, maxIdx] = max(profileVals);

% Calculate 6dB drop: 
halfMaxVal = maxVal - 6 ;  

% Find left and right indices where profile crosses half-max
% Left side: search from start to peak
leftIdx = find(profileVals(1:maxIdx) <= halfMaxVal, 1, 'last');
if isempty(leftIdx)
    leftIdx = 1;  % If never drops below half-max, use the start
end

% Right side: search from peak to end
rightIdx = find(profileVals(maxIdx:end) <= halfMaxVal, 1, 'first') + maxIdx - 1;
if isempty(rightIdx)
    rightIdx = nSamples;  % If never drops below half-max, use the end
end

% fwhm_angle_deg = (maxIdx-leftIdx) + (rightIdx-maxIdx); % wrong calculation 

% Calculate the physical distance in matrix units
% Get the actual coordinates of left and right points
rowLeft = rowLine(leftIdx);
colLeft = colLine(leftIdx);
rowRight = rowLine(rightIdx);
colRight = colLine(rightIdx);

% Distance between the two half-max points
fwhm_distance = sqrt((rowRight - rowLeft)^2 + (colRight - colLeft)^2);

% Calculate the distance from (61,30) to the peak
distance_to_peak = norm(peakPoint - startPoint);

% Calculate FWHM angle in degrees:
fwhm_angle_rad = 2 * atan(fwhm_distance / (2 * distance_to_peak));
fwhm_angle_deg = rad2deg(fwhm_angle_rad);

% Display results
fprintf('\n--- FWHM Analysis ---\n');
fprintf('Maximum value: %.4f\n', maxVal);
fprintf('Half-max value (-6dB): %.4f\n', halfMaxVal);
fprintf('FWHM distance (matrix units): %.2f\n', fwhm_distance);
fprintf('Distance from (25,25) to peak: %.2f\n', distance_to_peak);
fprintf('FWHM angle: %.2f degrees\n', fwhm_angle_deg);

%% 8. Visualize FWHM on the profile plot
figure(gcf);  % Use current figure
subplot(1,2,2)
hold on;
% Mark the maximum point
plot(maxIdx, maxVal, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
% Draw horizontal line at half-max level
yline(halfMaxVal, 'r--', 'LineWidth', 1.5, 'Label', '-6dB level');
% Mark left and right half-max points
plot(leftIdx, profileVals(leftIdx), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
plot(rightIdx, profileVals(rightIdx), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
% Add vertical lines to show FWHM width
plot([leftIdx, leftIdx], [0, halfMaxVal], 'g--', 'LineWidth', 1);
plot([rightIdx, rightIdx], [0, halfMaxVal], 'g--', 'LineWidth', 1);
legend('Profile', 'Peak', '-6dB level', 'FWHM points', 'Location', 'best');
title(sprintf('FWHM = %.2f°', fwhm_angle_deg));
hold off;

% Optional: Show FWHM on the matrix visualization
subplot(1,2,1)
hold on;
plot(colLeft, rowLeft, 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
plot(colRight, rowRight, 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
% legend({'Start (25,25)', 'Peak', 'Perpendicular cross-section', 'FWHM points'}, ...
%        'TextColor', 'w', 'Location', 'best');
hold off;

%% 9. Alignment of beams for the polar plot:

% shifting all vectors so the max is at zero:
profileVals_matrix = [R4S1_peak_values ;R4S2_peak_values;R5S1_peak_values;R5S2_peak_values;R6S1_peak_values];

load("profileVals_matrix.mat");

beamMat = zeros(size(profileVals_matrix));

for sessIndx = 1:size(profileVals_matrix,1)
    beamAvg = profileVals_matrix(sessIndx,:);
    alignedVec = zeros(1,length(profileVals_matrix));

    for ii = 1:size(beamAvg,1)
        [~, idxMax] = max(beamAvg(ii,:));

        % Shift so max moves to index corresponding to 0 deg
        shiftAmount = 251 - idxMax;   % index 251 corresponds to 0 deg
        alignedVec(ii,:) = circshift(beamAvg(ii,:), shiftAmount);
    end

    beamMat(sessIndx,:) = alignedVec;
end

%% plot C: polar plot showing the beam width

load("aligned_beamMat.mat");

fwhm_angles = [154.13	129.44	130.26	125.25	134.71];
n = length(aligned_beams);

figure();
for ii=1:size(aligned_beams,1)

    alignedBeam = aligned_beams(ii,:) - min(aligned_beams(ii,:)); % so all numbers are positive
    alignedBeam = alignedBeam / max(alignedBeam); % normalize the beams so all maximums are at 1db
    
    % Find the peak index
    [~, peakIdx] = max(alignedBeam);
    
    % Convert FWHM angle to number of samples on each side
    total_angle = 360; % degrees represented by full array
    samples_per_degree = n / total_angle;
    half_width_samples = round((fwhm_angles(ii) / 2) * samples_per_degree);
    
    % Define the index range around the peak
    startIdx = peakIdx - half_width_samples;
    endIdx   = peakIdx + half_width_samples;
    
    % Extract the FWHM window (handle wrap-around)
    indices = mod((startIdx:endIdx) - 1, n) + 1;
    
    % Build angle axis (in radians) for the selected indices
    theta = (indices - 1) * (2*pi / n);
    
    polarplot(theta, alignedBeam(indices));
    hold on;

end
ax = gca;
ax.RLim = [0, 1.2];
ax.RTick = [0, 0.2, 0.4, 0.6, 0.8, 1];
legend('R4S1','R4S2','R5S1','R5S2','R1S2')
title('Beams clipped to FWHM');

exportgraphics(gcf, 'beams.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');








