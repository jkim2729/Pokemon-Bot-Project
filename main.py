import pandas as pd
from pkm_alt_class import Pokemon_s, team_generator
import random
from poke_env.environment.move import Move
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
import numpy as np
import pettingzoo
from pettingzoo import ParallelEnv
import functools
import pickle
import math
from copy import copy
stats = pd.read_csv('pokemon-bst-totals.csv')
stats['Name'] = stats['Name'].apply(str.lower)
stats.set_index('Name',inplace=True)
with open('pkm_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
move_dict = loaded_data['move_dict']
pkm_moves_dict = loaded_data['pkm_moves_dict']
dex_nums = loaded_data['dex_nums']
rev_dex_nums = {v: k for k, v in dex_nums.items()}
status_boosts_dict = loaded_data['status_boosts_dict']
rev_status_boosts_dict = {v: k for k, v in status_boosts_dict.items()}
alt_status_dict = loaded_data['alt_status_dict']
status_dict = loaded_data['status_dict']
pokemon_list = loaded_data['pokemon_list']
partial_trapped_dict = loaded_data['partial_trapped_dict']
class Pokebot_Gen1(ParallelEnv):
    metadata = {'name':'pokemon_gen_1_v1'}

    action_spaces = {'p1':Discrete(11),'p2':Discrete(11)}
    def __init__(self):

        '''
        We represent this with 2 dimensional array. 12 is number of pokemon
        Each array contains required information
        We represent it as [Pokemon Species (dex number), Opponent or Player Pokemon(0 represents player and 1 represents opponent), Currently Active(0 is false, 1 is true), 
        Current Status Condition (Note:sleep turns are included here),Additional Status Conditions (Confusion/Leech Seed), Partial Trapped Turns, Toxic Counter,
          Current Health, Max Health, Base Speed, Current Speed, Current Special, Base Attack, Current Attack, Current Defense, Level, Speed boosts, Special boosts, 
          Attack boosts, Defense boosts, Reflect-LightScreen, Move 1 Name, Move 1 PP, Move 2 Name, Move 2 PP, 
          Move 3 Name, Move 3 PP, Move 4 Name, Move 4 PP,prev_dmg (used for counter)]
          
        '''
        
        self._p1_data = np.zeros(shape=(12,30),dtype=np.uint16)
        self._p2_data = np.zeros(shape=(12,30),dtype=np.uint16)
        self._finished = None
        self._battle_array = np.zeros(shape=(12,30),dtype=np.uint16)
        self.possible_agents = ['p1','p2']
        self._turn = 0
        self._active_index = (-1,-1)

    @property
    def p1_data(self):
        return self._p1_data
    
    def __set_p1_data_element__(self,row,column,new_data):
        self._p1_data[row,column] = new_data
    
    def __set_p1_data_row__(self,row,new_data):
        self._p1_data[row] = new_data
    
    def __set_p1_data_column__(self,column,new_data):
        self._p1_data[:,column] = new_data

    def __set_p1_data_full__(self,new_data):
        self._p1_data = new_data
    @property
    def p2_data(self):
        return self._p2_data
    
    def __set_p2_data_element__(self,row,column,new_data):
        self._p2_data[row,column] = new_data
    
    def __set_p2_data_row__(self,row,new_data):
        self._p2_data[row] = new_data
    
    def __set_p2_data_column__(self,column,new_data):
        self._p2_data[:,column] = new_data

    def __set_p2_data_full__(self,new_data):
        self._p2_data = new_data

    @property
    def battle_array(self):
        return self._battle_array
    
    def __set_battle_array_element__(self,row,column,new_data):
        self._battle_array[row,column] = new_data
    
    def __set_battle_array_row__(self,row,new_data):
        self._battle_array[row] = new_data
    
    def __set_battle_array_column__(self,column,new_data):
        self._battle_array[:,column] = new_data

    def __set_battle_array_full__(self,new_data):
        self._battle_array = new_data

    @property
    def active_index(self):
        return self._active_index
    @active_index.setter
    def active_index(self,new_index):
        self._active_index =  new_index

    @property
    def finished(self):
        return self._finished
    
    @finished.setter
    def finished(self,new_state:bool):
        self._finished = new_state

    @property
    def turn(self):
        return self._turn
    @turn.setter
    def turn(self,updated_turn):
        self._turn = updated_turn

    def create_pkm_array(self,spcs = 0, player = 0, active = 0, curr_status = 0, add_status = 0, partial_trap = 0, toxic_counter = 1, curr_health = 1000, max_health = 1000, base_speed = 0,
                         curr_speed = 0, curr_special = 0, base_attack = 0, curr_attack = 0, curr_defense = 0, level = 0, speed_boosts = 0, special_boosts = 0, attack_boosts = 0, defense_boosts = 0,
                         reflghtscreen = 0, mv1n = 1, mv1pp = 100, mv2n = 1, mv2pp = 100, mv3n = 1, mv3pp = 100, mv4n = 1,mv4pp = 100):
        
        return np.array([spcs,player,active,curr_status,add_status,partial_trap,toxic_counter,curr_health,max_health,base_speed,curr_speed,curr_special,base_attack,curr_attack,
                         curr_defense,level,speed_boosts,special_boosts,attack_boosts,defense_boosts,reflghtscreen,mv1n,mv1pp,mv2n,mv2pp,mv3n,mv3pp,mv4n,mv4pp],dtype= np.uint16)

    def reset(self, seed = None, options = {}):

        super().reset(seed=seed, options=options)

        if 'p1team' in options: #pokemon team was passed directly
            p1_mon_list = options['p1team']
        else:
            p1_mon_list = team_generator(seed_num=seed)
        if 'p2team' in options:
            p2_mon_list = options['p2team']
        else:
            p2_mon_list = team_generator(seed_num=seed)
        master_array = np.array([])
        p1_array = np.array([])
        for pkm in p1_mon_list:
            move_ids_pp = []
            for move in [pkm.move_1, pkm.move_2, pkm.move_3, pkm.move_4]:
                if isinstance(move, Move):
                    move_ids_pp.extend([move_dict[move.id], move.current_pp])
                else:
                    move_ids_pp.extend([move_dict['nomove'], 0])
            pkm_p1_array = self.create_pkm_array(spcs=dex_nums[pkm.species],curr_health=pkm.hp,max_health=pkm.hp,base_speed=pkm.speed,curr_speed=pkm.speed,
                                                 curr_special=pkm.special,base_attack=pkm.attack,curr_attack=pkm.attack,
                                                 curr_defense=pkm.defense,level=100,mv1n=move_ids_pp[0],mv1pp=[1],mv2n=move_ids_pp[2],mv2pp=move_ids_pp[3],mv3n=move_ids_pp[4],
                                                 mv3pp=move_ids_pp[5],mv4n=move_ids_pp[6],mv4pp=move_ids_pp[7])

            
            p1_array = np.vstack((p1_array, pkm_p1_array)) if p1_array.size > 0 else pkm_p1_array
        
        while len(p1_array) < 6:
            filler_pkm = self.create_pkm_array(curr_status=2,curr_health=0)
            p1_array = np.vstack((p1_array, filler_pkm))
        
        p2_array = np.array([])
        for pkm in p2_mon_list:
            move_ids_pp = []
            for move in [pkm.move_1, pkm.move_2, pkm.move_3, pkm.move_4]:
                if isinstance(move, Move):
                    move_ids_pp.extend([move_dict[move.id], move.current_pp])
                else:
                    move_ids_pp.extend([move_dict['nomove'], 0])
            pkm_p2_array = self.create_pkm_array(spcs=dex_nums[pkm.species],player=1,curr_health=pkm.hp,max_health=pkm.hp,base_speed=pkm.speed,curr_speed=pkm.speed,
                                                 curr_special=pkm.special,base_attack=pkm.attack,curr_attack=pkm.attack,
                                                 curr_defense=pkm.defense,level=100,mv1n=move_ids_pp[0],mv1pp=[1],mv2n=move_ids_pp[2],mv2pp=move_ids_pp[3],mv3n=move_ids_pp[4],
                                                 mv3pp=move_ids_pp[5],mv4n=move_ids_pp[6],mv4pp=move_ids_pp[7])
            
            p2_array = np.vstack((p2_array, pkm_p2_array)) if p2_array.size > 0 else pkm_p2_array
        
        while len(p2_array) < 6:
            filler_pkm = self.create_pkm_array(player=1,curr_status=2,curr_health=0)
            p2_array = np.vstack((p2_array, filler_pkm))
        
        master_array = np.vstack((p1_array, p2_array))
        for i in range(6):
            pkm_p1_array = self.create_pkm_array()

            p2_array = np.vstack((p2_array,pkm_p1_array))

            pkm_p2_array = self.create_pkm_array(player=1)
            p1_array = np.vstack((p1_array,pkm_p2_array))
        self.p1_data = p1_array
        self.p2_data = p2_array
        self.agents = copy(self.possible_agents)
        self.battle_array = master_array
        self.finished = False
        self.turn = 0 
        self.active_index = (-1,-1)
        observations = {'p1':p1_array,'p2':p2_array}

        info = {a: {} for a in self.agents}
        return tuple(observations,info)

    def new_mon(self,player,row): #corresponds to player of new pokemon, so player 1 seeing a new player 2 mon would be player 2
        mon_name = rev_dex_nums[self.battle_array[row,0]]
        if player == 0:
            
            self.p1_data[6+row,[0,7,8,9,10,11,12,13,14]] = [self.battle_array[row,0],stats.loc[mon_name]['HP_Total'],stats.loc[mon_name]['HP_Total'],stats.loc[mon_name]['Speed_Total'],
                                                               stats.loc[mon_name]['Speed_Total'],stats.loc[mon_name]['Special_Total'],stats.loc[mon_name]['Attack_Total'],
                                                    stats.loc[mon_name]['Attack_Total'],stats.loc[mon_name]['Defense_Total']]
        if player == 1:
            self.p2_data[6+row,[0,7,8,9,10,11,12,13,14]] = [self.battle_array[row,0],stats.loc[mon_name]['HP_Total'],stats.loc[mon_name]['HP_Total'],stats.loc[mon_name]['Speed_Total'],
                                                      stats.loc[mon_name]['Speed_Total'],stats.loc[mon_name]['Special_Total'],stats.loc[mon_name]['Attack_Total'],
                                                    stats.loc[mon_name]['Attack_Total'],stats.loc[mon_name]['Defense_Total']]

    # def use_move(self,player,row,opponent):

    def step(self,actions):
        
        ### initialize variables (observation will come at the end)
        truncations = {a: False for a in self.agents}
        terminations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        p1_action = actions['p1']
        p2_action = actions['p2']

        ###    
        if self.turn == 0:
            p1_active = p1_action-4
            p2_active = p2_action-4
            self.active_index = (p1_active,p2_active)
            self.p1_data[p1_active,2] = 1
            self.p2_data[p2_active,2] = 1
            self.battle_array[p1_active,2] = 1
            self.battle_array[p2_active+6,2] = 1
            self.new_mon(0,p2_active)
            self.new_mon(1,p1_active)
            observations = {'p1':self.p1_data,'p2':self._p2_data}

            return observations, rewards, terminations, truncations, infos
        
        ###Battle mechanics, unique situations are when both pokemon stay in and use a move, both switch, or one switches and the other attacks
        

        p1_active = self.active_index[0]
        p2_active = self.active_index[1]        
        if p1_action<=3 and p2_action<=3: #both players make a move
            p1_active_array = self.battle_array[p1_active]
            p1_eff_speed = math.floor(p1_active_array[9])
            p2_active_array = self.battle_array[p2_active]
            p2_eff_speed = math.floor(p2_active_array[9])
            if p1_eff_speed>p2_eff_speed:
                first_mover = 'player1'
            elif p1_eff_speed<p2_eff_speed:
                first_mover = 'player2'
            else:
                speed_tb = random.random()          
                if speed_tb>0.5:
                    first_mover = 'player1'
                else:
                    first_mover = 'player2'
            # if first_mover == 'player1': 

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        high_values = np.array([255,1,1,31,15,15,63,1023,1023,1023,1023,1023,1023,1023,1023,127,15,15,15,15,3,255,127,255,127,255,127,255,127,1023])
        high_matrix = np.tile(high_values, (12, 1)) 
        observation_spaces = {'p1':Box(
                    low=0,  
                    high=high_matrix,  
                    shape=(12, 30), 
                    dtype=np.uint16
                ), 'p2':Box(
                    low=0,  
                    high=high_matrix,  
                    shape=(12, 30), 
                    dtype=np.uint16
                )}
        return observation_spaces
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(11)
from pettingzoo.test import parallel_api_test
env = Pokebot_Gen1()
parallel_api_test(env,num_cycles=1)