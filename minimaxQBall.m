% minimax-Q learning for two agents learning together

%% initial settings
iteration = 2000;
% settings for the game
% for state 1 if play 1 choose action 2, it can get more instant reward,
% if it choose action2, it will be more likely to stay at state 1
% Reward1 = {[-0.75,-0.8;-0.9,-0.6],[0,0.9;0.85,0.1]};
%Reward1 = {[-0.75,-0.8;-0.9,-0.6],[0.1,0.85;0.9,0]};
Reward1 = {[-0.75,-0.8;-0.9,-2],[4,0.85;0.9,0.6]};

Q1 = {[1,1;1,1],[1,1;1,1]};
V1 = [1,1];
pi1 = {[1/2,1/2],[1/2,1/2]};
Q2 = {[1,1;1,1],[1,1;1,1]};
V2 = [1,1];
pi2 = {[1/2,1/2],[1/2,1/2]};
alpha = 1;
decay = 0.995;
% transfer = {[0.4,0.75;0.6,0.2],[0.25,0.75;0.6,0.2]};%possibility to go to state1
transfer = {[0.9,0.15;0.1,0.95],1-[0.9,0.15;0.1,0.95]};
currentState = 1;
currentAction = 1;%action for play1
currentOpponent = 1;%action for play2
gamma = 0.9;
rewardResult = zeros(iteration,1);
rewardResult2 = zeros(iteration,1);
vResult = zeros(iteration,2);
vResult2 = zeros(iteration,2);
pi1Result = zeros(iteration,2);
pi2Result = zeros(iteration,2);
pi21Result = zeros(iteration,2);
pi22Result = zeros(iteration,2);
%% learning process
for k = 1:iteration
    
    currentAction = (rand>pi1{currentState}(1))*1+1; %select action based on pi
    currentOpponent = (rand>pi2{currentState}(1))*1+1; %select opponent action randomly
    nextState = 1+(rand>transfer{currentState}(currentAction,currentOpponent))*1;% next state
    % update Q of play 1 based on previous Q, the V value of next
    % state, the state is observed instead of estimated. 
    Q1{currentState}(currentAction,currentOpponent) = (1-alpha)*Q1{currentState}(currentAction,currentOpponent)...
        +alpha*(Reward1{currentState}(currentAction,currentOpponent)+gamma*V1(nextState)); % update Q1
    Q2{currentState}(currentAction,currentOpponent) = (1-alpha)*Q2{currentState}(currentAction,currentOpponent)...
        +alpha*(-Reward1{currentState}(currentAction,currentOpponent)+gamma*V2(nextState)); % update Q2
    % minimax optimization for play1
    fun = @(piOpt)[-piOpt*Q1{currentState}(:,1);-piOpt*Q1{currentState}(:,2)];
    Aeq = [1,1];
    beq = [1];
    lb = [0,0];
    ub = [1,1];
    x0 = pi1{currentState};
    A = [];
    b =[];
    
    [PiCandidate,VCandidate] = fminimax(fun,x0,A,b,Aeq,beq,lb,ub);
    V1(currentState) = min(-VCandidate);
    pi1{currentState} = PiCandidate;
    
    % minimax optimization for play2
    
    fun = @(piOpt)[-piOpt*Q2{currentState}(1,:)';-piOpt*Q2{currentState}(2,:)'];
    Aeq = [1,1];
    beq = [1];
    lb = [0,0];
    ub = [1,1];
    x0 = pi2{currentState};
    A = [];
    b =[];
    
    [PiCandidate,VCandidate] = fminimax(fun,x0,A,b,Aeq,beq,lb,ub);
    V2(currentState) = min(-VCandidate);
    pi2{currentState} = PiCandidate;
    %
    alpha = alpha*decay;
    if k == 1
        rewardResult(k,1) = (Reward1{currentState}(currentAction,currentOpponent));
    else
        rewardResult(k,1) = rewardResult(k-1,1)*((k-1)/k)+(1/k)*(Reward1{currentState}(currentAction,currentOpponent));
    end
    if k == 1
        rewardResult(k,2) = (-Reward1{currentState}(currentAction,currentOpponent));
    else
        rewardResult(k,2) = rewardResult(k-1,1)*((k-1)/k)+(1/k)*(-Reward1{currentState}(currentAction,currentOpponent));
    end
    currentState = nextState;
    pi1Result(k,:) = pi1{1};
    pi2Result(k,:) = pi1{2};
    pi21Result(k,:) = pi2{1};
    pi22Result(k,:) = pi2{2};
    vResult(k,:) = V1;
    vResult2(k,:) = V2;
end

%% plot
figure
subplot(2,1,1)
plot(vResult(:,1))
title('player 1 V for state 1')
subplot(2,1,2)
plot(vResult(:,2))
title('player 1 V for state 2')
sgtitle('player 1 V')

figure
subplot(2,1,1)
plot(vResult2(:,1))
title('player 2 V for state 1')
subplot(2,1,2)
plot(vResult2(:,2))
title('player 2 V for state 2')
sgtitle('player 2 V')

figure
subplot(2,2,1)
plot(pi1Result(:,1))
title('choose action 1 in state 1')
subplot(2,2,2)
plot(pi1Result(:,2))
title('choose action 2 in state 1')
subplot(2,2,3)
plot(pi2Result(:,1))
title('choose action 1 in state 2')
subplot(2,2,4)
plot(pi2Result(:,2))
title('choose action 2 in state 2')
sgtitle('player 1 \pi')

figure
subplot(2,2,1)
plot(pi21Result(:,1))
title('choose action 1 in state 1')
subplot(2,2,2)
plot(pi21Result(:,2))
title('choose action 2 in state 1')
subplot(2,2,3)
plot(pi22Result(:,1))
title('choose action 1 in state 2')
subplot(2,2,4)
plot(pi22Result(:,2))
title('choose action 2 in state 2')
sgtitle('player 2 \pi')