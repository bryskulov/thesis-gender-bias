% Authors: Bakbergen Ryskulov
% Macroeconometrics
% Wold.m function
% Inputs:
    % companion = companion matrix associated to the VAR process
    % n = number of dependent variables
    % horizon = horizon
% Output:
    % irf = matrix which contains the Input Response Functions for each
    % variable, shock and horizon

function [irf] = Wold(companion, n, horizon)
    % Empty 3D matrix to store results
    irf = zeros(n, n, horizon);
    for j = 1 : horizon
       power = companion^(j-1);
       irf(:,:,j) = power(1:n, 1:n);
    end
end