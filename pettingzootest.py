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
from damage import damage,select_dmg_value,confusion_dmg
from copy import copy
from poke_env.environment import AbstractBattle
from poke_env.player.player import Player


with open('pkm_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
move_dict = loaded_data['move_dict']
rev_move_dict = {v: k for k, v in move_dict.items()}
pkm_mvs_dict = loaded_data['pkm_mvs_dict']
dex_nums = loaded_data['dex_nums']
rev_dex_nums = {v: k for k, v in dex_nums.items()}

class Pokebot_Gen1(Player):
    def create_pkm_array(self,spcs = 0, player = 0, active = 0, curr_status = 0, add_status = 0, partial_trap = 0, toxic_counter = 1, curr_health = 1000, 
                         curr_speed = 0,curr_attack = 0, curr_defense = 0, level = 0, 
                         speed_boosts = 0, special_boosts = 0, attack_boosts = 0, defense_boosts = 0, accuracy_boosts = 0,
                         reflghtscreen = 0, mv1n = 1, mv1pp = 100, mv2n = 1, mv2pp = 100, mv3n = 1, mv3pp = 100, mv4n = 1,mv4pp = 100,prev_dmg = 0,substitute_hp = 0):
        
        return np.array([spcs,player,active,curr_status,add_status,partial_trap,toxic_counter,curr_health,curr_speed,curr_attack,
                         curr_defense,level,speed_boosts,special_boosts,attack_boosts,defense_boosts,accuracy_boosts,reflghtscreen,
                         mv1n,mv1pp,mv2n,mv2pp,mv3n,mv3pp,mv4n,mv4pp,prev_dmg,substitute_hp],dtype= np.uint16)
    

    def embed_battle(self, battle: AbstractBattle):
        player_team = battle.team
        opponent_team = battle.opponent_team
        player_num = 0
        
        for mon in player_team:

            pkm_array = self.create_pkm_array
            pkm_array[]

        return (player_team,player_mon,opponent_team,opponent_mon)
    




    def choose_move(self, battle):
    
        return self.choose_random_move(battle)