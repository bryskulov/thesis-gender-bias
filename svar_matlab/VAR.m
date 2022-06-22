% Author: Bakbergen Ryskulov
% Macroeconometrics
% VAR.m : function which creates VAR() process with p lags, n
% variables and the option for the constant

% Inputs:
    % Data = our data (dependent variable) --> matrix
    % p = number of lags
    % constant = option for constant (1 --> yes, 0 --> no)
% Outputs:
    % y = dependent variable
    % x = independent variable
    % beta_hat = beta_hat coefficients' computed using OLS
    % companion = companion matrix associated to the VAR process

function [y, x, beta_hat, companion] = VAR(data, p, constant)
    % Save number of periods and variables
    [T, n] = size(data);
    % We have p_lags. It means that our dependent variable y loses p observations 
    % and it must starts at p+1
    y = data(p+1:end,:);
    % Dependent variable : x
    if (constant == 1)
        % Define x
        x = ones(T-p,n*p+1);
        for j = 1 : p
            x(:,(j-1)*n+2:j*n+1) = data(p-j+1:end-j,:);                   
        end
        % Compute OLS
        beta_hat = (x'*x)\x'*y;
        % Create vector of zeros
        vect_zeros = zeros(n*(p-1),n);
        % Create identity matrix
        identity = eye(n*(p-1));
        % Transpose beta_hat
        beta_trans = beta_hat';
        coeff = beta_trans(:,2:end);
        % Define companion matrix
        companion = [coeff;
                        identity, vect_zeros];
        
    else
         x = ones(T-p,n*p);
        for j = 1 : p
            x(:,(j-1)*n+1:j*n) = data(p-j+1:end-j,:);                   
        end
        % Compute OLS
        beta_hat = (x'*x)\x'*y;
        % Create vector of zeros
        vect_zeros = zeros(n*(p-1),n);
        % Create identity matrix
        identity = eye(n*(p-1));
        % Transpose beta_hat
        beta_trans = beta_hat';
        coeff = beta_trans;
        % Define companion matrix
        companion = [coeff; identity, vect_zeros];
    end  
end