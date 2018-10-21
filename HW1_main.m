clear;clc;
%parpool(6)
%% Set up parameters
bbeta               = 0.96;
T                   = 10;
rrho                = 0.9;                                  % persistence of human capital
ggamma              = 0.7;                                  % exponent of education term
w                   = 1;                                    % wage
eepsilon.values     = [-0.0673,-0.0336,0,0.0336,0.0673];    % human capital shocks
eepsilon.transition = [0.9727 0.0273 0 0 0; ...
                       0.0041 0.9806 0.0153 0 0; 
                       0 0.0082 0.9836 0.0082 0;
                       0 0 0.0153 0.9806 0.0041;
                       0 0 0 0.0273 0.9727];
r.values            = [0.04 0.05 0.06];
r.transition        = [0.9 0.1 0; 0.05 0.9 0.05; 0 0.1 0.9];
%%%%%%%%%%%%%%%%%%%%%%
% Version 1: turn off uncertainty (r=0.05 and eepsilon = 0)
r.values = 0.05;
r.transition = 1;
eepsilon.values = 0;
eepsilon.transition = 1;
%%%%%%%%%%%%%%%%%%%%%%
na                  = 200;
nh                  = 30;
nr                  = length(r.values);
neepsilon           = length(eepsilon.values);
V                   = zeros(T,na,nh,nr,neepsilon);
policy.a            = zeros(T,na,nh,nr,neepsilon);
policy.h            = zeros(T,na,nh,nr,neepsilon);

%% Create grid for endogenous state variables
% a grid
amin                = -9;
amax                = 4;
agrid               = linspace(amin,amax,na);

% h grid
hmin                = 0.1;
hmax                = 3;
hgrid               = linspace(hmin,hmax,nh);

%% Backward recursion
tic;
tempV = zeros(na*nh*nr*neepsilon,1);
tempA = nan(na*nh*nr*neepsilon,1);
tempH = nan(na*nh*nr*neepsilon,1);

% Period T
for ih = 1 : nh
    hcurr = hgrid(ih);
    for ia = 1 : na
        acurr = agrid(ia);
        for ir = 1 : nr
            rcurr = r.values(ir);
            cons = 1/2*w^2*hcurr^2+(1+rcurr)*acurr;
            
            if cons <= 0
                V(T,ia,ih,ir,:) = -1e5;
            else
                V(T,ia,ih,ir,:) = log(cons);
            end
            
        end
    end
end

% Periods T-1 to 1
for age = T-1 : -1 : 1
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
        
        VV = -1e3; 
        hChoice = nan; 
        aChoice = nan;
        transitionProb = kron(eepsilon.transition(ieepsilon,:),r.transition(ir,:)');
        transitionProb = reshape(transitionProb,neepsilon*nr,1)';
        
        % Loop over states in the next period
        for ihp = 1 : nh
            
            education = (hgrid(ihp)-0.9*humanCapital)^(1/(0.7+eduShock));
            labor = w*humanCapital-education;
            totalIncome = w*humanCapital*labor+(1+interestRate)*asset;
            % Human capital at t+1 cannot go below 0.9h and labor >= 0
            if hgrid(ihp) < 0.9*humanCapital || labor < 0
                continue
            end
            
            for iap = 1 : na                                               
                consumption = totalIncome - agrid(iap);
                if consumption < 0
                    continue
                end
                Vnext = reshape(V(age+1,iap,ihp,:,:),nr*neepsilon,1);                

                const = consumption - (education+labor)^2/2;
                
                if const <= 0 
                    utility = -1e5;
                else
                    utility = log(const) + bbeta*Vnext;
                end
                
                if utility >= VV
                    VV = utility;
                    hChoice = ihp;
                    aChoice = iap;                    
                end
                utility = 0;
                
            end        
        end
        
        tempV(ind) = VV;
        tempA(ind) = aChoice;
        tempH(ind) = hChoice;
    end
    
    for ind = 1 : (na*nh*nr*neepsilon)
        ia = floor(mod(ind-0.05,na))+1;
        ih = mod(floor((ind-0.05)/na),nh)+1;
        ir = mod(floor((ind-0.05)/(na*nh)),nr)+1;
        ieepsilon = mod(floor((ind-0.05)/(na*nh*nr)),neepsilon)+1;
        
        V(age,ia,ih,ir,ieepsilon) = tempV(ind);
        policy.a(age,ia,ih,ir,ieepsilon) = tempA(ind);
        policy.h(age,ia,ih,ir,ieepsilon) = tempH(ind);
    end
    
    finish = toc;
    disp(['Age: ', num2str(age), '. Time: ', num2str(finish),' seconds'])
end

%% Generate paths from policy functions
[~,ia] = min(abs(agrid));
[~,ih] = min(abs(hgrid-1));
ir = 1;
ieepsilon = ceil(neepsilon/2);
aPath = [agrid(ia) nan(1,T-2)];
hPath = [hgrid(ih) nan(1,T-2)];
for age = 1 : T-2   
    ia = policy.a(age,ia,ih,ir,ieepsilon);
    ih = policy.h(age,ia,ih,ir,ieepsilon);
    aPath(age+1) = agrid(ia);
    hPath(age+1) = hgrid(ih);
end
plot(aPath);