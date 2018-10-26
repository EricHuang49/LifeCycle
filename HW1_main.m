clear;clc;
warning off
parpool(2)
%% Set up parameters
bbeta               = 0.96;
T                   = 10;
rrho                = 0.9;                                  % persistence of human capital
ggamma              = 0.7;                                  % exponent of education term
w                   = 1;                                    % wage
eepsilon.values     = [-0.0673,-0.0336,0,0.0336,0.0673];    % human capital shocks
eepsilon.transition = [0.9727 0.0273 0.0000 0.0000 0.0000; ...
                       0.0041 0.9806 0.0153 0.0000 0.0000; 
                       0.0000 0.0082 0.9836 0.0082 0.0000;
                       0.0000 0.0000 0.0153 0.9806 0.0041;
                       0.0000 0.0000 0.0000 0.0273 0.9727];
r.values            = [0.04 0.05 0.06];
r.transition        = [0.9 0.1 0; 0.05 0.9 0.05; 0 0.1 0.9];
na                  = 250;  
nh                  = 100;
nr                  = length(r.values);
neepsilon           = length(eepsilon.values);

%% Create grid for endogenous state variables
% a grid
amin                = 0;
amax                = 3;
agrid               = linspace(amin,amax,na);

% h grid
hmin                = 0.6;
hmax                = 7;
hgrid               = linspace(hmin,hmax,nh);

%% Initialize value and policy functions
V                   = zeros(T,na,nh,nr,neepsilon);
policy.a            = zeros(T-1,na,nh,nr,neepsilon);        % assets next period
policy.h            = zeros(T-1,na,nh,nr,neepsilon);        % human capital next period
policy.l            = zeros(T,na,nh,nr,neepsilon);          % labor supply
policy.c            = zeros(T,na,nh,nr,neepsilon);          % consumption
policy.e            = zeros(T,na,nh,nr,neepsilon);          % education

%% Backward recursion
tempV = zeros(na*nh*nr*neepsilon,1);
tempA = nan(na*nh*nr*neepsilon,1);
tempH = nan(na*nh*nr*neepsilon,1);
tempL = nan(na*nh*nr*neepsilon,1);
tempC = nan(na*nh*nr*neepsilon,1);
tempE = nan(na*nh*nr*neepsilon,1);

% Period T
for ih = 1 : nh
    hcurr = hgrid(ih);
    for ia = 1 : na
        acurr = agrid(ia);
        for ir = 1 : nr
            rcurr = r.values(ir);
            const = 1/2*w^2*hcurr^2+(1+rcurr)*acurr;
            
            if const <= 0
                V(T,ia,ih,ir,:) = -1e5;
            else
                V(T,ia,ih,ir,:) = log(const);
            end
            
        end
    end
    policy.l(T,:,ih,:,:) = w*hcurr;
    policy.c(T,:,ih,:,:) = w*hcurr*policy.l(T,:,ih,:,:) + (1+rcurr)*acurr;
end

tic;
% Periods T-1 to 1
for age = T-1 : -1 : 1
    
    % First calculate the expected continuation value to speed up code
    VnextExpected = zeros(na,nh,nr,neepsilon);                  % a and h are T+1 states; r and eepsilon are current 
    Vnext = squeeze(V(age+1,:,:,:,:));
    for ir = 1 : nr
        for ieepsilon = 1 : neepsilon
            Vp = reshape(Vnext,[],neepsilon);
            Vp = Vp * eepsilon.transition(ieepsilon,:)';
            %Vp = reshape(Vp,na,nh,nr);
            Vp = reshape(Vp,[],nr);
            Vp = Vp * r.transition(ir,:)';
            VnextExpected(:,:,ir,ieepsilon) = reshape(Vp,na,nh);
        end
    end
    
    % Iterate over current states
    parfor ind = 1 : (na*nh*nr*neepsilon)
        ia = floor(mod(ind-0.05,na))+1;
        ih = mod(floor((ind-0.05)/na),nh)+1;
        ir = mod(floor((ind-0.05)/(na*nh)),nr)+1;
        ieepsilon = mod(floor((ind-0.05)/(na*nh*nr)),neepsilon)+1;
        
        % Current states
        asset = agrid(ia);
        humanCapital = hgrid(ih);
        interestRate = r.values(ir);
        eduShock = eepsilon.values(ieepsilon);
        
        VV = -1e5; 
        hChoice = nan; aChoice = nan; lChoice = nan; cChoice = nan; eChoice = nan;
        
        % Iterate over states in the next period
        for ihp = 1 : nh
            education = ((hgrid(ihp)-rrho*humanCapital)/exp(eduShock))^(1/ggamma);
            labor = w*humanCapital-education;
            
            % Restrictions on human capital at t+1 and labor choice
            if education < 0 || labor < 0
                continue
            end
                        
            totalIncome = w*humanCapital*labor+(1+interestRate)*asset;
            utilityPrevA = -1e5;
            
            for iap = 1 : na                                               
                consumption = totalIncome - agrid(iap);
                
                % If consumption turns negative, stop searching on asset
                if consumption < 0
                    break
                end

                const = consumption - (education+labor)^2/2;     
                
                % If the term inside log utility <= 0, stop searching on asset
                if const <= 0 
                    break
                else
                    utility = log(const) + bbeta*VnextExpected(iap,ihp,ir,ieepsilon);
                end
               
                % Update value function if we find a bigger value
                if utility >= VV
                    VV = utility;
                    hChoice = ihp;
                    aChoice = iap;      
                    lChoice = labor;
                    cChoice = consumption;
                    eChoice = education;
                end
            end
        end
        
        tempV(ind) = VV;
        tempA(ind) = aChoice;
        tempH(ind) = hChoice;
        tempL(ind) = lChoice;
        tempC(ind) = cChoice;
        tempE(ind) = eChoice;
    end
    
    % Fill into the policy functions
    for ind = 1 : (na*nh*nr*neepsilon)
        ia = floor(mod(ind-0.05,na))+1;
        ih = mod(floor((ind-0.05)/na),nh)+1;
        ir = mod(floor((ind-0.05)/(na*nh)),nr)+1;
        ieepsilon = mod(floor((ind-0.05)/(na*nh*nr)),neepsilon)+1;
        
        V(age,ia,ih,ir,ieepsilon) = tempV(ind);
        policy.a(age,ia,ih,ir,ieepsilon) = tempA(ind);
        policy.h(age,ia,ih,ir,ieepsilon) = tempH(ind);
        policy.l(age,ia,ih,ir,ieepsilon) = tempL(ind);
        policy.e(age,ia,ih,ir,ieepsilon) = tempE(ind);
        policy.c(age,ia,ih,ir,ieepsilon) = tempC(ind);
    end
    
    finish = toc;
    disp(['Age: ', num2str(age), '. Time: ', num2str(finish),' seconds'])
end

%% Simulate paths from policy functions
% Initialize the paths
path.savings         = zeros(T,15);
path.asset           = zeros(T,15);
path.consumption     = zeros(T,15);
path.labor           = zeros(T,15);
path.education       = zeros(T,15);
path.humanCapital    = ones(T,15);

% Generate Markov Chains for interest rate and human capital shock
rng(1);
markov.eduShock     = simulate(dtmc(eepsilon.transition),T-1,'X0',3*ones(1,neepsilon));
markov.interestRate = [];
for amountsimulation = 1: (15/3)
    markov.interestRate = horzcat(markov.interestRate,  ...
        simulate(dtmc(r.transition),T-1, 'X0', [1 1 1]));
end

% Generate the paths using policy functions
for j = 1 : size(markov.eduShock,2)
    for age = 1 : T
        [~,ia] = min(abs(agrid-path.asset(age,j)));
        [~,ih] = min(abs(hgrid-path.humanCapital(age,j)));
        ir = markov.interestRate(age,j);
        ieepsilon = markov.eduShock(age,j);
        
        path.labor(age,j) = policy.l(age,ia,ih,ir,ieepsilon);
        path.education(age,j) = w*path.humanCapital(age,j) - path.labor(age,j);
        path.education(age,j) = policy.e(age,ia,ih,ir,ieepsilon);
        
        if age < T
            path.savings(age,j) = agrid(policy.a(age,ia,ih,ir,ieepsilon)) - path.asset(age,j);
            path.asset(age+1,j) = agrid(policy.a(age,ia,ih,ir,ieepsilon));
            path.humanCapital(age+1,j) = hgrid(policy.h(age,ia,ih,ir,ieepsilon));
            path.consumption(age,j) = policy.c(age,ia,ih,ir,ieepsilon);          
        else
            path.savings(T,j) = 0;
            path.consumption(T,j) = w*path.humanCapital(age,j)*path.labor(age,j) + ...
                (1+r.values(ir))*path.asset(age,j);
        end
        
    end
end

% Plot the simulated paths
subplot(2,3,1)
plot(1:T,path.savings)
title('Savings')
subplot(2,3,2)
plot(1:T,path.asset)
title('Asset')
subplot(2,3,3)
plot(1:T,path.consumption)
title('Consumption')
subplot(2,3,4)
plot(1:T,path.labor)
title('Labor')
subplot(2,3,5)
plot(1:T,path.education)
title('Education')
subplot(2,3,6)
plot(1:T,path.humanCapital)
title('Human Capital')