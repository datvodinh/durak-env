import numpy as np
from numba import njit
import numba
from numba.typed import List
import sys
value = np.array(['2','3','4','5','6','7','8','9','10','J','Q','K','A'])
type_card = np.array(['♤','♧','♢','♡'])
# @njit
def get_card(id):
        if id <52:
            v = id%13
            t = id//13
            return f'{value[v]}{type_card[t]}'
        else:
            return 'skip'
@njit
def initEnv():
    env = np.zeros(80)
    card = np.arange(52)#card
    np.random.shuffle(card)
    for i in range(4):
        env[card[i*8:(i+1)*8]] = i+1
    env[52] = card[-1] #trump suit card
    env[53] = 0 #mode: attack or defense
    env[54:57] = [1,3,4]
    env[57]= 1 #num of people choose attack this round
    env[58] = 2 #player_id defending
    env[59] = 0 #index player attack in env[54:57]
    env[60:80] = card[32:52] #card on deck
    return env
@njit
def getStateSize():
    return 163
@njit
def getAgentState(env):
    state = np.zeros(getStateSize())
    if env[53]==0:
        player_id = env[54:57][int(env[59])] #attack player id
    elif env[53]==1:
        player_id = env[58] #defend player id
    state[0:52][np.where(env[0:52]==player_id)] = 1 # card player hold
    state[52:104][np.where(env[0:52]==6)] = 1 #all card defender defend this round
    state[104:156][np.where(env[0:52]==5)] = 1#card have to defend this round
    if env[53]==1:
        state[156:158] = [0,1] # attack, defend
    elif env[53]==0:
        state[156:158] = [1,0]
    state[158:162][int(env[52])//13] = 1 #trump suit
    state[162] = len(np.where(env[0:52]==0)[0]) #num card on deck
    return state

@njit
def getActionSize():
    return 53
@njit
def getDefenseCard(state):
    card = np.zeros(52)
    idx = np.argmax(state[158:162]) #trump suit: 0:spade,1:club,2:diamond,3:heart
    card_def_id = np.argmax(state[104:156])
    if card_def_id//13!=idx:#card have to defend not a trump card
        card[13*idx:13*(idx+1)][np.where(state[13*idx:13*(idx+1)]==1)] = 1 # trump card on hand
        card[card_def_id+1:13*(card_def_id//13+1)][np.where(state[card_def_id+1:13*(card_def_id//13+1)]==1)] = 1 #same type card, higher value on hand.
    else:#card have to defend is a trump card
        card[card_def_id+1:13*(idx+1)][np.where(state[card_def_id+1:13*(idx+1)]==1)] = 1 #higher value trump card only.
    return card
    
@njit
def getAttackCard(state):
    card = np.zeros(52)
    card_on_board = np.where(state[52:104]==1)[0]
    card_value_on_board = card_on_board % 13 #value of that card (ex: 4 diamond is 4)
    card_on_hand = np.where(state[0:52]==1)[0] #card on player's hand
    for c in card_on_hand:
        if c%13 in card_value_on_board:
            card[c] = 1
    return card
@njit
def getValidAction(state):
    list_action = np.zeros(getActionSize())
    #attack
    if state[156]==1 and np.sum(state[52:104])==0: #main attacker, defender have nothing to defend yet.
        list_action[0:52] = state[0:52]
    elif state[156]==1 and np.sum(state[52:104])!=0:#side attacker, attack only card with same value on the defend board( 4 heart on hand if have 4 spade on board)
        list_action[0:52] = getAttackCard(state)
        list_action[52] = 1
    #defense
    if state[157]==1:#defender
        list_action[0:52] = getDefenseCard(state)
        list_action[52] = 1
    return list_action
@njit
def drawCard(env):
    turn_draw_card = np.zeros(4)
    turn_draw_card[np.array([0,2,3])] = env[54:57] #attack player,main attack draw first.
    turn_draw_card[1] = env[58] #defend player draw second.
    for p_id in turn_draw_card: #draw card
        num_card_on_deck = len(np.where(env[0:52]==0)[0])#num cards left on deck
        if num_card_on_deck > 0:
            num_card_player = len(np.where(env[0:52]==p_id)[0])
            if num_card_player < 8:
                num_card_need = 8 - num_card_player
                if num_card_on_deck >= num_card_need:
                    env[env[60:80].astype(np.int64)[20-num_card_on_deck:20-num_card_on_deck+num_card_need]] = p_id
                else:
                    env[env[60:80].astype(np.int64)[20-num_card_on_deck:]] = p_id
    return env
@njit
def changeAttackPlayer(env): #change the defender and attacker
    if env[58]==1:
        env[54:57] = [4,2,3]
    elif env[58]==2:
        env[54:57] = [1,3,4]
    elif env[58]==3:
        env[54:57] = [2,4,1]
    elif env[58]==4:
        env[54:57] = [3,1,2]
@njit
def stepEnv(action,env):
    if action == 52:#skip
        if env[53] == 1: #defense
            env[0:52][np.where(env[0:52]==5)] = env[58] #Attacker hold all card
            env[0:52][np.where(env[0:52]==6)] = env[58] #Attacker hold all card
            env = drawCard(env) #draw card
            env[58] = (env[58]+2)%4 if env[58] > 2 else env[58]+2 #change defend player
            changeAttackPlayer(env) #change attack player
            env[53] = 0 #reset mode: attack
            env[59] = 0
        elif env[53] == 0:#attack
            env[57] += 1 #num attacker skip this round
            env[59]  = (env[59]+1)%3
            if env[57] == 3:#all attacker skip this round
                env[0:52][np.where(env[0:52]==5)] = -1#Thrown away card
                env[0:52][np.where(env[0:52]==6)] = -1#Thrown away card
                env = drawCard(env) #draw card
                env[58] = 1 if env[58]==4 else env[58]+1 #change defend player
                changeAttackPlayer(env) #change attack players
                env[57] = 0
                env[59] = 0
                
    else:#attack or defend any card
        if env[53] == 1:#defense
            env[0:52][np.where(env[0:52]==5)] = 6 #defense this card successful
            env[action] = 6
            env[53] = 0#change mode: attack
        elif env[53] == 0: #attack
            env[action] = 5 #this card have to defend
            env[53] = 1 #change mode: defend
            env[57] = 0 #change num player attack skip turn to 0
            env[59]  = (env[59]+1)%3 #change attack player
            
    # return env


@njit
def checkEnded(env):
    if len(np.where(env[0:52]==0)[0])==0:#if no card left on deck
        list_win = []
        turn_draw_card = np.zeros(4)
        turn_draw_card[np.array([0,2,3])] = env[54:57]
        turn_draw_card[1] = env[58]
        for p_id in turn_draw_card:
            if len(np.where(env[0:52]==p_id)[0]) == 0: #if player have no card left
                list_win.append(p_id)
            else:
                pass
        if len(list_win)>0:
            return int(list_win[0]-1)
        else:
            return -1
    return -1

def one_game(listAgent,perData):
    env = initEnv()
    tempData = []
    for _ in range(4):
        dataOnePlayer = List()
        dataOnePlayer.append(np.array([[0.]]))
        tempData.append(dataOnePlayer)
    print(env[:53])
    winner = -1
    turn = 0
    while True:
        for i in range(1,5):
            print(f'P{i}:',end=" ")
            for card in np.where(env[0:52]==i)[0]:
                print(get_card(card),end=" ")
            print("")
        print(f'Turn {turn}; Trump card: {get_card(int(env[52]))}; Defend id: {env[58]}; Attack id :{env[54:57][int(env[59])]};',end=" ")
        turn +=1
        if env[53]==1:#defense
            pIdx = int(env[58] - 1)
            print(f'Player: {pIdx+1} is defending;',end=" ")
            

        else:#attack
            pIdx = int(env[54:57][int(env[59])] - 1)
            print(f'Player: {pIdx+1} is attacking;',end=" ")
        action, tempData[pIdx], perData = listAgent[pIdx](getAgentState(env), tempData[pIdx], perData)
        print(f'Action index: {get_card(action)};',end=" ")
        if env[53]==1:
            if action==52:
                print("Defend Fail")
            else:
                print("Defend successful")
        else:
            if action==52:
                print("Player skip")
            else:
                print("Player attack")
        stepEnv(action, env)
        # print(env[:53])
        winner = checkEnded(env)
        if winner != -1:
            break
    return winner, perData
@njit
def numba_one_game(p0, p1, p2, p3, perData, pIdOrder):
    env = initEnv()
    tempData = []
    for _ in range(4):
        dataOnePlayer = List()
        dataOnePlayer.append(np.array([[0.]]))
        tempData.append(dataOnePlayer)
    
    winner = -1
    while True:
        if env[53]==0:
            pIdx = int(env[58] - 1)
        else:
            pIdx = int(env[54:57][int(env[59])] - 1)
        try:
            if pIdOrder[pIdx] == 0:
                action, tempData[pIdx], perData = p0(getAgentState(env), tempData[pIdx], perData)
            elif pIdOrder[pIdx] == 1:
                action, tempData[pIdx], perData = p1(getAgentState(env), tempData[pIdx], perData)
            elif pIdOrder[pIdx] == 2:
                action, tempData[pIdx], perData = p2(getAgentState(env), tempData[pIdx], perData)
            elif pIdOrder[pIdx] == 3:
                action, tempData[pIdx], perData = p3(getAgentState(env), tempData[pIdx], perData)
        except:
            print(list(env))
            break
        stepEnv(action, env)
        winner = checkEnded(env)
        if winner != -1:
            break
    
    return winner, perData
        
# @njit
def normal_main(listAgent, times, perData):
    numWin = np.full(5, 0)
    pIdOrder = np.arange(4)
    for _ in range(times):
        np.random.shuffle(pIdOrder)
        shuffledListAgent = [listAgent[i] for i in pIdOrder]
        winner, perData = one_game(shuffledListAgent, perData)
        if winner == -1:
            numWin[4] += 1
        else:
            numWin[pIdOrder[winner]] += 1
    return numWin, perData
@njit
def numba_main(p0, p1, p2, p3, times, perData):
    numWin = np.full(5, 0)
    pIdOrder = np.arange(4)
    for _ in range(times):
        np.random.shuffle(pIdOrder)
        winner, perData = numba_one_game(p0, p1, p2, p3, perData, pIdOrder)
        if winner == -1:
            numWin[4] += 1
        else:
            numWin[pIdOrder[winner]] += 1
    return numWin, perData

@njit
def ramdom_player(state,temp,per):
    list_action  = np.where(getValidAction(state)==1)[0]
    action = np.random.choice(list_action)
    # print(list(state))
    # print(f'List action: {list_action}',end=" ")
    return action,temp,per
main([ramdom_player,ramdom_player,ramdom_player,ramdom_player],1,0)
# numbaMain(ramdom_player,ramdom_player,ramdom_player,ramdom_player,500000,0)
