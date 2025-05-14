import pandas as pd
from pkm_alt_class import Pokemon_s, team_generator
import random
from poke_env.environment.move import Move
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
import numpy as np
import pettingzoo
from itertools import zip_longest
from pettingzoo import ParallelEnv
import functools
import pickle
import math
from damage import damage,select_dmg_value,confusion_dmg
from copy import copy
from poke_env.environment import AbstractBattle
from poke_env.player.player import Player
stats = pd.read_csv('pokemon-bst-totals.csv')
stats['Name'] = stats['Name'].apply(str.lower)
stats.set_index('Name',inplace=True)
with open('pkm_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
move_dict = loaded_data['move_dict']
rev_move_dict = {v: k for k, v in move_dict.items()}
pkm_mvs_dict = loaded_data['pkm_mvs_dict']
dex_nums = loaded_data['dex_nums']
rev_dex_nums = {v: k for k, v in dex_nums.items()}
status_boosts_dict = loaded_data['status_boosts_dict']
rev_status_boosts_dict = {v: k for k, v in status_boosts_dict.items()}
alt_status_dict = loaded_data['alt_status_dict']
status_dict = loaded_data['status_dict']
pokemon_list = loaded_data['pokemon_list']
partial_trapped_dict = loaded_data['partial_trapped_dict']
boosts_dict = loaded_data['boost_dict']
TypeChart = loaded_data['TypeChart']
full_acc_moves = loaded_data['full_acc_moves']
pokemon_types = loaded_data['pokemon_types']




class Pokebot_Gen1(Player):
    def __init__(self, account_configuration = None, *, avatar = None, battle_format = "gen9randombattle", log_level = None, max_concurrent_battles = 1, accept_open_team_sheet = False, save_replays = False, server_configuration = None, start_timer_on_battle_start = False, start_listening = True, ping_interval = 20, ping_timeout = 20, team = None):
        super().__init__(account_configuration, avatar=avatar, battle_format=battle_format, log_level=log_level, max_concurrent_battles=max_concurrent_battles, accept_open_team_sheet=accept_open_team_sheet, save_replays=save_replays, server_configuration=server_configuration, start_timer_on_battle_start=start_timer_on_battle_start, start_listening=start_listening, ping_interval=ping_interval, ping_timeout=ping_timeout, team=team)
        self.seed_dict = {}
        self.player_num = None
        self.opponent_num = None
        self.prev_embed_battle = None
        self.player_active_index = 0
        self.opponent_active_index = 0
    def get_state(self, battle: AbstractBattle):
        player_team = battle.team
        opponent_team = battle.opponent_team
        if battle.turn == 1:
            for mon in player_team:
                mon_tuple = (0,dex_nums[mon.species])
                self.seed_dict[mon_tuple] = 1
            for mon in opponent_team:
                mon_tuple = (1,dex_nums[mon.species])
                self.seed_dict[mon_tuple] = 1
        player_mon = battle.active_pokemon
        opponent_mon = battle.opponent_active_pokemon
        return (player_team,player_mon,opponent_team,opponent_mon)

    def embed_battle(self,battle:AbstractBattle):

        ### partial trapping, toxic, confusion, leech seed, counter substitute are going to be implemented later
        player_team = battle.team
        
        opponent_team = battle.opponent_team
        partial_trap = 0
        toxic_counter = 0
        alt_status = 0
        sub_hp = 0 
        player_mon = battle.active_pokemon
        opponent_mon = battle.opponent_active_pokemon
        opp_status = opponent_mon.status.value
        opp_para_flag = False
        opp_brn_flag = False
        play_para_flag = False
        play_brn_flag = False
        if opp_status == 1:
            opp_brn_flag = True
        elif opp_status == 4:
            opp_para_flag = True
        player_status = player_mon.status.value
        if player_status == 1:
            play_brn_flag = True
        elif player_status == 4:
            play_para_flag = True
        
        player_array = np.zeros(shape=(12,26),dtype=np.uint16)
        observations = battle.observations
        
        if not self.player_num:
            for ev in observations.events:
                if str(self.username) in ev and 'player' in ev:
                    if 'p1' in ev or 'p1a' in ev or 'p1b' in ev:
                        self.player_num = 1
                        self.opponent_num = 2
                    else:
                        self.player_num = 1
                        self.opponent_num = 2

        index = 0
        zipped = zip_longest(player_team,opponent_team,fillvalue=0)
        for pmon,omon in zipped:
            pspecies = pmon.species
            ospecies = omon.species
            pdex_num = dex_nums[pspecies]
            odex_num = dex_nums[ospecies]
            pteam_num = 0
            oteam_num = 1
            events = observations.events
            mover = None
            if pmon.active:
                pactive_num = 1
                self.player_active_index = index
                if self.prev_embed_battle[index][8] == 0:
                    p_base_speed = stats[pspecies]['Speed_Total']
                    p_current_speed = stats[pspecies]['Speed_Total']
                if self.prev_embed_battle[index][10] == 0:
                    p_base_special = stats[pspecies]['Special_Total']
                if self.prev_embed_battle[index][11] == 0:
                    p_base_attack = stats[pspecies]['Attack_Total']
                    p_current_attack = o_base_attack
                if self.prev_embed_battle[index][13] == 0:
                    p_base_defense = stats[pspecies]['Defense_Total']

            else:
                pactive_num = 0
            if omon.active:
                oactive_num = 1
                self.opponent_active_index = index
                if self.prev_embed_battle[index+6][8] == 0:
                    o_base_speed = stats[ospecies]['Speed_Total']
                    o_current_speed = stats[ospecies]['Speed_Total']
                if self.prev_embed_battle[index+6][10] == 0:
                    o_base_special = stats[ospecies]['Special_Total']
                if self.prev_embed_battle[index+6][11] == 0:
                    o_base_attack = stats[ospecies]['Attack_Total']
                    o_current_attack = o_base_attack
                if self.prev_embed_battle[index+6][13] == 0:
                    o_base_defense = stats[ospecies]['Defense_Total']




            else:
                oactive_num = 0

            p_speed_boosts = self.prev_embed_battle[index][17]
            p_special_boosts = self.prev_embed_battle[index][18]
            p_attack_boosts = self.prev_embed_battle[index][19]
            p_defense_boosts = self.prev_embed_battle[index][20]
            p_accuracy_boosts = self.prev_embed_battle[index][21]
            o_speed_boosts = self.prev_embed_battle[index+6][17]
            o_special_boosts = self.prev_embed_battle[index+6][18]
            o_attack_boosts = self.prev_embed_battle[index+6][19]
            o_defense_boosts = self.prev_embed_battle[index+6][20]
            o_accuracy_boosts = self.prev_embed_battle[index+6][21]
        

            for e in events:
                event_name = e[1]
                if event_name == 'move':
                    mover = e[2][1]
                    if mover == 1:
                        non_mover = 2
                    else:
                        non_mover = 1
                if event_name == 'boost' or event_name == '-unboost':
                    target = e[2][1]
                    if mover == self.opponent_num and target == self.player_num:
                        if 'spe' in e:
                            p_current_speed = p_base_speed*boosts_dict[p_speed_boosts]
                            if play_para_flag:
                                p_current_speed = p_current_speed//4
                        elif 'atk' in e:
                            p_current_attack = p_base_attack*boosts_dict[p_attack_boosts]
                            if play_brn_flag:
                                p_current_attack = p_current_attack//2
                    elif mover == self.opponent_num and target == mover:
                        if 'spe' in e:
                            o_current_speed = o_base_speed*boosts_dict[o_speed_boosts]
                            if play_para_flag:
                                p_current_speed = p_current_speed//4
                        elif 'atk' in e:
                            o_current_attack = o_base_attack*boosts_dict[o_attack_boosts]
                            if play_brn_flag:
                                p_current_attack = p_current_attack//2
                    elif mover == self.player_num and target == self.opponent_num:
                        if 'spe' in e:
                            o_current_speed = o_base_speed*boosts_dict[o_speed_boosts]
                            if opp_para_flag:
                                o_current_speed = o_current_speed//4
                        elif 'atk' in e:
                            o_current_attack = o_base_attack*boosts_dict[o_attack_boosts]
                            if opp_brn_flag:
                                o_current_attack = o_current_attack//2
                    elif mover == self.player_num and target == mover:
                        if 'spe' in e:
                            p_current_speed = p_base_speed*boosts_dict[p_speed_boosts]
                            if opp_para_flag:
                                o_current_speed = o_current_speed//4
                        elif 'atk' in e:
                            p_current_attack = p_base_attack*boosts_dict[o_attack_boosts]
                            if opp_brn_flag:
                                o_current_attack = o_current_attack//2                                                         
                                





            pstat_condition = pmon.status.value
            if pstat_condition == 7:
                pstat_condition = 13
            elif pstat_condition == 6:
                pstat_condition+= (pmon.status_counter-1) # need to verify what they count as turn 0
            
            php_stat = pmon.current_hp

            ostat_condition = omon.status.value
            if ostat_condition == 7:
                ostat_condition = 13
            elif ostat_condition == 6:
                ostat_condition+= (omon.status_counter-1) # need to verify what they count as turn 0
            
            ohp_stat = omon.current_hp


            index+=1

            


    def choose_move(self, battle):

        return self.choose_random_move(battle)

class Pokebot_Gen1_Environment(ParallelEnv):
    metadata = {'name':'pokemon_gen_1_v1'}

    action_spaces = {'p1':Discrete(11),'p2':Discrete(11)}

    def __init__(self):
        
        '''
        We represent this with 2 dimensional array. 12 is number of pokemon
        Each array contains required information
        We represent it as [Pokemon Species (dex number), Opponent or Player Pokemon(0 represents player and 1 represents opponent), Currently Active(0 is false, 1 is true), 
        Current Status Condition (Note:sleep turns are included here),Additional Status Conditions (Confusion/Leech Seed), Partial Trapped Turns, Toxic Counter,
          Current Health, Max Health, Base Speed, Current Speed, Current Special, Base Attack, Current Attack, Current Defense, Level, Speed boosts, Special boosts, 
          Attack boosts, Defense boosts, Accuracy boosts, Reflect-LightScreen, Move 1 Name, Move 1 PP, Move 2 Name, Move 2 PP, 
          Move 3 Name, Move 3 PP, Move 4 Name, Move 4 PP,prev_dmg (used for counter),substitute_hp (0 if no substitute is in place)]
        '''
        
        self._p1_data = np.zeros(shape=(12,26),dtype=np.uint16)
        self._p2_data = np.zeros(shape=(12,26),dtype=np.uint16)
        self._p1_battle_object = None
        self._p2_battle_oject = None

    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)

        if 'p1team' in options: #pokemon team was passed directly
            p1_team = options['p1team']
        else:
            p1_team = team_generator(seed_num=seed)
        if 'p2team' in options:
            p2_team = options['p2team']
        else:
            p2_team = team_generator(seed_num=seed)



    @property
    def p1_data(self):
        return self._p1_data
    
    @property
    def p2_data(self):
        return self._p2_data
        
