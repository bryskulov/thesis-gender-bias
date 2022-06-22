% Author: Bakbergen Ryskulov
% bootstrap.m : function which compute bootstrap in order to compute the
% confidence band of the impulse response function

% Inputs:
    % Data = our data (dependent variable) --> matrix
    % n = number variables
    % p = number of lags
    % constant = option for constant (1 --> yes, 0 --> no)
    % horizon = number of horizon
    % up = upper bound level --> number --> in our case = 84
    % lb = lower bound level --> number --> in our case = 16
% Outputs:
    % irf = 3D matrix which contains the IRF for the Wold representation
    % cirf = 3D matrix which contains the IRF for the Cholesky representation
    % low_irf = 3D matrix which contains the lower-bound of the IRF for
    % Wold
    % upp_irf = 3D matrix which contains the upper-bound of the IRF for
    % Wold
    % low_irf_chol = 3D matrix which contains the lower-bound of the IRF for
    % Cholesky
    % upp_irf_chol = 3D matrix which contains the upper-bound of the IRF for
    % Cholesky

function [irf, cirf, low_irf, upp_irf, low_irf_chol, upp_irf_chol]=bootstrap(data, n, p, constant, horizon, up, lb)
        % Recall the VAR function to find the companion matrix which will
        % be used to construct the Wold representation
        [y, x, beta_true, companion] = VAR(data, p, constant);
        % Estimate the IRF for Wold representation
        [irf] = Wold(companion, n, horizon);        
        % Now we have beta_true, we must compute the errors(fitted OLS residuals)
        errors = [y - x * beta_true];
        % Estimate the IRF for Cholesky (it is the only one which can be 
        % interpreted because it guarantees both IDENTIFICATION and EXOGENEITY)
        % Compute omega (Variance-Covariance matrix of the errors)
        omega_true = [errors' * errors] ./ size(errors, 1);
        % Cholesky decomposition
        B0 = chol(omega_true)';
        % Define irf_chol
        cirf = zeros(size(irf));
        % Compute IRF for cholesky
        for i = 1:size(irf,3)
            cirf(:,:,i) = irf(:,:,i)*B0;
        end
        % Save number of rows of dependent variable
        rows = size(y, 1);
        % Total repetition for bootstrap
        reps = 1000;
        % Define the 4D-matrix which contains the irf of the bootstrap
        irf_store = ones(n,n,horizon,reps);
        % Define the 4D-matrix which contains the irf of the bootstrap for
        % cholesky
        cirf_store = ones(n,n,horizon,reps);
        
        % Compute Bootstrap
        for k = 1 : reps
            % Let us define a matrix which contains the "fake" dependent
            % variable for each iteration of the bootstrap
            X_fake = ones(size(x));
            Y_fake = ones(size(y));
    
            for j = 1 : rows 
                % Pick randomly an integer which allows me to pick the error in
                % that position
                position = randi([1,rows]);
                % Save error in that particular position
                rand_err = errors(position, :);      
                % Create the fake dataset
                if (j == 1)
                     X_fake(j,:) = x(j,:);
                else
                    X_fake(j,:) = [ones(constant,1) Y_fake(j-1,:) X_fake(j-1,constant+1:end-n)];
                end
                Y_fake(j,:) = X_fake(j,:) * beta_true + rand_err;           
            end
        % Now we estimate the beta of the fake data for each repetition, to
        % construct the companion matrix. Thus, we recall again the VAR function
        [junk, junk, beta_hat, companion_reps] = VAR(Y_fake, p, constant);
        % We need the Wold coefficients, let us recall Wold function
        [irf_reps] = Wold(companion_reps, n, horizon);
        % Store the input response function for each repetition
        irf_store(:,:,:,k) = irf_reps;
        % Now we have beta_hat, we must compute the errors(fitted OLS residuals)
        errors_hat = [y - x * beta_hat];
        % Estimate the IRF for Cholesky (it is the only one which can be 
        % interpreted because it guarantees both IDENTIFICATION and EXOGENEITY)
        % Compute omega (Variance-Covariance matrix of the errors)
        omega_hat = [errors_hat' * errors_hat] ./ size(errors_hat, 1);
        % Cholesky decomposition
        B0_hat = chol(omega_hat)';
        % Compute IRF for cholesky
        for i = 1:size(irf_store,3)
            cirf_store(:,:,i,k) = irf_store(:,:,i,k)*B0_hat;
        end    
    end
        
    % We need to store the IRF for the upper bound and the lower bound of the
    % confidence interval for each variable and shock
    % Wold
        low_irf = ones(n,n,horizon);
        upp_irf = ones(n,n,horizon);
    % Cholesky
        low_irf_chol = ones(n,n,horizon);
        upp_irf_chol = ones(n,n,horizon);
    
    % Repeat for each variable and shock of Wold representation
    % For each Variable
    for i = 1 : n
        % For each shock
        for j = 1 : n
            low_irf(i,j,:) = prctile(irf_store(i,j,:,:), lb, 4);
            upp_irf(i,j,:) = prctile(irf_store(i,j,:,:), up, 4);
        end
    end
    
    % Repeat for each variable and shock of Cholesky representation
    % For each Variable
    for i = 1 : n
        % For each shock
        for j = 1 : n
            low_irf_chol(i,j,:) = prctile(cirf_store(i,j,:,:), lb, 4);
            upp_irf_chol(i,j,:) = prctile(cirf_store(i,j,:,:), up, 4);
        end
    end
end