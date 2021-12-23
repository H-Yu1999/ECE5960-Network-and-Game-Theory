% minimax learning for two agents learning together

%% initial settings
iteration = 1000;
% settings for the game
% for state 1 if play 1 choose action 2, it can get more instant reward,
% if it choose action2, it will be more likely to stay at state 1
Reward1 = {[1,0;0,2],[-1,0;0,-2]};
Q1 = {[1,1;1,1],[1,1;1,1]};
V1 = [1,1];
pi1 = {[1/2,1/2],[1/2,1/2]};
Q2 = {[1,1;1,1],[1,1;1,1]};
V2 = [1,1];
pi2 = {[1/2,1/2],[1/2,1/2]};
alpha = 1;
decay = 0.999;
% transfer = {[0.4,0.75;0.6,0.2],[0.25,0.75;0.6,0.2]};%possibility to go to state1
transfer = {[0.4,0.75;0.6,0.2],1-[0.4,0.75;0.6,0.2]};%possibility to go to state1
currentState = 1;
currentAction = 1;%action for play1
currentOpponent = 1;%action for play2
gamma = 0.9;
%% some matrix to record the precess
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
    
    currentAction = (rand>(pi1{currentState}(1)))*1+1; %select action based on pi
    currentOpponent = (rand>(pi2{currentState}(1)))*1+1; %select opponent action randomly
    nextState = 1+(rand>transfer{currentState}(currentAction,currentOpponent))*1;% next state
    % update Q of play 1 based on previous Q, the V value of potential next
    % state(estimated)
    Q1{currentState}(currentAction,currentOpponent) = (1-alpha)*Q1{currentState}(currentAction,currentOpponent)...
        +alpha*(Reward1{currentState}(currentAction,currentOpponent)+...
        gamma*transfer{currentState}(currentAction,currentOpponent)*V1(1)+...
        gamma*(1-transfer{currentState}(currentAction,currentOpponent))*V1(2)); 
    Q2{currentState}(currentAction,currentOpponent) = (1-alpha)*Q2{currentState}(currentAction,currentOpponent)...
        +alpha*(-Reward1{currentState}(currentAction,currentOpponent)+...
        gamma*transfer{currentState}(currentAction,currentOpponent)*V2(1)+...
        gamma*(1-transfer{currentState}(currentAction,currentOpponent))*V2(2)); % update Q
    
    % minimax optimization for play1
    fun = @(piOpt)[-piOpt*Q1{currentState}(:,1);-piOpt*Q1{currentState}(:,2)];%use minus reward to convert minimax to maxmini 
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
    
    fun = @(piOpt)[-piOpt*Q2{currentState}(1,:)';-piOpt*Q2{currentState}(2,:)'];%use minus reward to convert minimax to maxmini 
    Aeq = [1,1];
    beq = [1];
    lb = [0,0];
    ub = [1,1];
    x0 = pi2{currentState};%starting point
    A = [];
    b =[];
    
    [PiCandidate,VCandidate] = fminimax(fun,x0,A,b,Aeq,beq,lb,ub);
    V2(currentState) = min(-VCandidate);
    pi2{currentState} = PiCandidate;
    %
    alpha = alpha*decay;
    % record the average reward of the two players
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
    currentState = nextState; %change to next state
    %record pi for two players
    pi1Result(k,:) = pi1{1}; 
    pi2Result(k,:) = pi1{2};
    pi21Result(k,:) = pi2{1};
    pi22Result(k,:) = pi2{2};
    %record V for two players
    vResult(k,:) = V1;
    vResult2(k,:) = V2;
end
%% plot
figure
subplot(2,1,1)
plot(vResult(:,1))
hold on
line([1,iteration],[0.6694,0.6694],'Color','r')
title('player 1 V for state 1')
subplot(2,1,2)
plot(vResult(:,2))
hold on
line([1,iteration],[-0.6690,-0.6690],'Color','r')
title('player 1 V for state 2')
sgtitle('player 1 V')

figure
subplot(2,1,1)
plot(vResult2(:,1))
hold on
line([1,iteration],[-0.6692,-0.6692],'Color','r')
title('player 2 V for state 1')
subplot(2,1,2)
plot(vResult2(:,2))
hold on
line([1,iteration],[0.6694,0.6694],'Color','r')
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