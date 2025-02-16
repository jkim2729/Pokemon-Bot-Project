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
from copy import copy

stats = pd.read_csv('pokemon-bst-totals.csv')
stats['Name'] = stats['Name'].apply(str.lower)
stats.set_index('Name',inplace=True)

class Pokebot_Gen1(ParallelEnv):
    metadata = {'name':'pokemon_gen_1_v1'}

    action_spaces = {'p1':Discrete(11),'p2':Discrete(11)}
    def __init__(self):

        '''
        We represent this with 2 dimensional array. 12 is number of pokemon
        Each array contains required information
        We represent it as [Pokemon Species (dex number), Opponent or Player Pokemon(0 represents player and 1 represents opponent), Currently Active(0 is false, 1 is true), 
        Current Status Condition (Note:sleep turns are included here),Additional Status Conditions (Confusion/Leech Seed), Partial Trapped Turns, Toxic Counter,
          Current Health, Max Health, Current Speed, Current Special, Current Attack, Current Defense, Level, Speed boosts, Special boosts, 
          Attack boosts, Defense boosts, Reflect-LightScreen, Move 1 Name, Move 1 PP, Move 2 Name, Move 2 PP, 
          Move 3 Name, Move 3 PP, Move 4 Name, Move 4 PP]
          
        '''
        
        self._p1_data = np.zeros(shape=(12,27),dtype=np.uint16)
        self._p2_data = np.zeros(shape=(12,27),dtype=np.uint16)
        self._finished = None
        self._battle_array = np.zeros(shape=(12,27),dtype=np.uint16)
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
            
            pkm_p1_array = np.array([
                dex_nums[pkm.species], 0, 0, 0, 0, 0, 1, pkm.hp, pkm.hp, pkm.speed,
                pkm.special, pkm.attack, pkm.defense, 100, 0, 0, 0, 0, 0, *move_ids_pp
            ], dtype=np.uint16)

            
            p1_array = np.vstack((p1_array, pkm_p1_array)) if p1_array.size > 0 else pkm_p1_array
        
        while len(p1_array) < 6:
            filler_pkm = np.array([0,0,0,2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0], dtype=np.uint16)
            p1_array = np.vstack((p1_array, filler_pkm))
        
        p2_array = np.array([])
        for pkm in p2_mon_list:
            move_ids_pp = []
            for move in [pkm.move_1, pkm.move_2, pkm.move_3, pkm.move_4]:
                if isinstance(move, Move):
                    move_ids_pp.extend([move_dict[move.id], move.current_pp])
                else:
                    move_ids_pp.extend([move_dict['nomove'], 0])
            
            pkm_p2_array = np.array([
                dex_nums[pkm.species], 1, 0, 0, 0, 0, 1, pkm.hp, pkm.hp, pkm.speed,
                pkm.special, pkm.attack, pkm.defense, 100, 0, 0, 0, 0, 0, *move_ids_pp
            ], dtype=np.uint16)
            
            p2_array = np.vstack((p2_array, pkm_p2_array)) if p2_array.size > 0 else pkm_p2_array
        
        while len(p2_array) < 6:
            filler_pkm = np.array([0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0], dtype=np.uint16)
            p2_array = np.vstack((p2_array, filler_pkm))
        
        master_array = np.vstack((p1_array, p2_array))
        for i in range(6):
            pkm_p2_array = np.array([0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
                                            ,1,0,1,0,1,0,1,0],dtype=np.uint16)
            p2_array = np.vstack((p1_array,pkm_p1_array))
            pkm_p1_array = np.array([0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
                                            ,1,0,1,0,1,0,1,0],dtype=np.uint16)
            p1_array = np.vstack((p1_array,pkm_p1_array))
        self.p1_data = p1_array
        self.p2_data = p2_array
        self.agents = copy(self.possible_agents)
        self.battle_array = master_array
        self.finished = False
        self.turn = 0 
        self.active_index = (-1,-1)
        observations = (p1_array,p2_array)

        info = {a: {} for a in self.agents}
        return (observations,info)

    def step(self,actions):
        p1_action = actions['p1']
        p2_action = actions['p2']


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        high_values = np.array([255,1,1,31,15,15,63,1023,1023,1023,1023,1023,1023,127,15,15,15,15,3,255,127,255,127,255,127,255,127])
        high_matrix = np.tile(high_values, (12, 1)) 
        observation_spaces = {'p1':Box(
                    low=0,  
                    high=high_matrix,  
                    shape=(12, 27), 
                    dtype=np.uint16
                ), 'p2':Box(
                    low=0,  
                    high=high_matrix,  
                    shape=(12, 27), 
                    dtype=np.uint16
                )}
        return observation_spaces
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(11)