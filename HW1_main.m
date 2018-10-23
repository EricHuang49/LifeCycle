%clear;clc;
%parpool(6)
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
%%%%%%%%%%%%%%%%%%%%%%
% Version 1: turn off uncertainty (r=0.05 and eepsilon = 0)
% r.values = 0.05;
% r.transition = 1;
% eepsilon.values = 0;
% eepsilon.transition = 1;
%%%%%%%%%%%%%%%%%%%%%%
na                  = 10;
nh                  = 10;
nr                  = length(r.values);
neepsilon           = length(eepsilon.values);
V                   = zeros(T,na,nh,nr,neepsilon);
policy.a            = zeros(T-1,na,nh,nr,neepsilon);
policy.h            = zeros(T-1,na,nh,nr,neepsilon);
policy.l            = zeros(T,na,nh,nr,neepsilon);

%% Create grid for endogenous state variables
% a grid
amin                = 0;
amax                = 3;
agrid               = linspace(amin,amax,na);

% h grid
hmin                = 0.6;
hmax                = 7;
hgrid               = linspace(hmin,hmax,nh);

%% Backward recursion
tic;
tempV = zeros(na*nh*nr*neepsilon,1);
tempA = nan(na*nh*nr*neepsilon,1);
tempH = nan(na*nh*nr*neepsilon,1);
tempL = nan(na*nh*nr*neepsilon,1);

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
end

% Periods T-1 to 1
for age = T-1 : -1 : 1
    for ind = 1 : (na*nh*nr*neepsilon)
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
        %hChoice = nan; 
        %aChoice = nan;
        %transitionProb = kron(eepsilon.transition(ieepsilon,:),r.transition(ir,:)');
        %transitionProb = reshape(transitionProb,1,neepsilon*nr);
        
        % Loop over states in the next period
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
                % Restriction on consumption
                if consumption < 0
                    continue
                end
                %Vnext = reshape(V(age+1,iap,ihp,:,:),nr*neepsilon,1);                

                const = consumption - (education+labor)^2/2;                
                
                if const <= 0 
                    utility = -1e5;
                else
                    %VnextExpected = transitionProb.*squeeze(V(age+1,iap,ihp,:,:));
                    VnextExpected = 0;
                    for irp = 1 : nr
                        for ieepsilonp = 1 : neepsilon
                            VnextExpected = VnextExpected + ...
                                eepsilon.transition(ieepsilon,ieepsilonp) ...
                                *r.transition(ir,irp)*V(age+1,iap,ihp,irp,ieepsilonp);
                        end
                    end
                    
                    %utility = log(const) + bbeta*sum(VnextExpected(:));
                    utility = log(const) + bbeta*VnextExpected;
                end
                
                % Use concavity of the value function to speed up
                if utility < utilityPrevA
                    break
                else 
                    utilityPrevA = utility;
                end
                
                if utility >= VV
                    VV = utility;
                    hChoice = ihp;
                    aChoice = iap;      
                    lChoice = labor;
                end
                %utility = 0;
                
            end
        end
        
        tempV(ind) = VV;
        tempA(ind) = aChoice;
        tempH(ind) = hChoice;
        tempL(ind) = lChoice;
    end
    
    for ind = 1 : (na*nh*nr*neepsilon)
        ia = floor(mod(ind-0.05,na))+1;
        ih = mod(floor((ind-0.05)/na),nh)+1;
        ir = mod(floor((ind-0.05)/(na*nh)),nr)+1;
        ieepsilon = mod(floor((ind-0.05)/(na*nh*nr)),neepsilon)+1;
        
        V(age,ia,ih,ir,ieepsilon) = tempV(ind);
        policy.a(age,ia,ih,ir,ieepsilon) = tempA(ind);
        policy.h(age,ia,ih,ir,ieepsilon) = tempH(ind);
        policy.l(age,ia,ih,ir,ieepsilon) = tempL(ind);
    end
    
    finish = toc;
    disp(['Age: ', num2str(age), '. Time: ', num2str(finish),' seconds'])
end

%% Simulate paths from policy functions
savings         = zeros(T,75);
asset           = zeros(T,75);
consumption     = zeros(T,75);
labor           = zeros(T,75);
education       = zeros(T,75);
humanCapital    = ones(T,75);

rng(1);
markov.interestRate = simulate(dtmc(r.transition),T-1,'X0',25*ones(1,nr));
markov.eduShock     = simulate(dtmc(eepsilon.transition),T-1,'X0',15*ones(1,neepsilon));

for j = 1 : size(markov.eduShock,2)
    for age = 1 : T
        [~,ia] = min(abs(agrid-asset(age,j)));
        [~,ih] = min(abs(hgrid-humanCapital(age,j)));
        ir = markov.interestRate(age,j);
        ieepsilon = markov.eduShock(age,j);
        
        labor(age,j) = policy.l(age,ia,ih,ir,ieepsilon);
        education(age,j) = w*humanCapital(age,j) - labor(age,j);
        if age < T
            savings(age,j) = agrid(policy.a(age,ia,ih,ir,ieepsilon)) - asset(age,j);
            asset(age+1,j) = agrid(policy.a(age,ia,ih,ir,ieepsilon));
            humanCapital(age+1,j) = hgrid(policy.h(age,ia,ih,ir,ieepsilon));
            consumption(age,j) = w*humanCapital(age,j)*labor(age,j) + (1+r.values(ir))*asset(age,j) - asset(age+1,j);
        end
    end
    savings(T,j) = 0;
    consumption(T,j) = w*humanCapital(age,j)*labor(age,j) + (1+r.values(ir))*asset(age,j);
end
