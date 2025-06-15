import pandas as pd
from pkm_alt_class import Pokemon_s, team_generator_alt
import random
import time
from gymnasium.utils import seeding
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
import numpy as np
import pettingzoo
from itertools import zip_longest
from pettingzoo import ParallelEnv
import functools
import pickle
import math
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from numpy.random import Generator
from weakref import WeakKeyDictionary
import asyncio
from typing import Any, Awaitable, Dict, Generic, List, Optional, Tuple, TypeVar, Union
from damage import damage,select_dmg_value,confusion_dmg
from copy import copy
from poke_env import RandomPlayer
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    ForfeitBattleOrder,
)
from gymnasium.utils.env_checker import check_env
from poke_env.environment import AbstractBattle
from poke_env.player.player import Player
from poke_env.player import PokeEnv
from poke_env.player.env import _EnvPlayer
from poke_env.concurrency import POKE_LOOP, create_in_poke_loop
from poke_env.teambuilder.teambuilder import Teambuilder
import uuid
ItemType = TypeVar("ItemType")
ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")


team_1 = """
Alakazam  
Ability: No Ability  
- Psychic
- Seismic Toss
- Recover 
- Thunder Wave  

Snorlax  
Ability: No Ability  
- Body Slam  
- Hyper Beam  
- Earthquake  
- Self-Destruct  

Tauros  
Ability: No Ability  
- Blizzard  
- Body Slam  
- Earthquake  
- Hyper Beam  

Chansey  
Ability: No Ability  
- Ice Beam  
- Thunderbolt  
- Thunder Wave  
- Soft-Boiled  

Starmie  
Ability: No Ability  
- Thunder Wave  
- Psychic  
- Blizzard  
- Recover  

Exeggutor  
Ability: No Ability  
- Double-Edge  
- Explosion  
- Psychic  
- Sleep Powder  
"""
team_2 = """
Gengar  
Ability: none  
- Hypnosis  
- Psychic  
- Thunderbolt  
- Explosion  

Starmie  
Ability: none  
- Surf  
- Thunderbolt  
- Thunder Wave  
- Recover  

Zapdos  
Ability: Pressure  
- Thunderbolt  
- Drill Peck  
- Thunder Wave  
- Agility  

Chansey  
Ability: None
- Ice Beam  
- Thunderbolt  
- Thunder Wave  
- Soft-Boiled  

Snorlax  
Ability: None
- Body Slam  
- Hyper Beam 
- Earthquake  
- Self-Destruct  

Tauros  
Ability: None
- Body Slam  
- Hyper Beam  
- Blizzard  
- Earthquake  
"""
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


def create_pkm_array(spcs = 0, player = 1, active = 0, curr_status = 0, add_status = 0, partial_trap = 0, toxic_counter = 1, curr_health = 100, max_health = 100, 
                base_speed = 0,curr_speed = 0, curr_special = 0, base_attack = 0, curr_attack = 0, curr_defense = 0, level = 0, 
                speed_boosts = 7, special_boosts = 7, attack_boosts = 7, defense_boosts = 7, accuracy_boosts = 7,
                reflghtscreen = 0, mv1n = 1, mv1pp = 100, mv2n = 1, mv2pp = 100, mv3n = 1, mv3pp = 100, mv4n = 1,mv4pp = 100,prev_dmg=0,substitute_hp = 0):
            return np.array([spcs,player,active,curr_status,add_status,partial_trap,toxic_counter,curr_health,max_health,base_speed,curr_speed,curr_special,base_attack,curr_attack,
                        curr_defense,level,speed_boosts,special_boosts,attack_boosts,defense_boosts,accuracy_boosts,reflghtscreen,
                        mv1n,mv1pp,mv2n,mv2pp,mv3n,mv3pp,mv4n,mv4pp,prev_dmg,substitute_hp],dtype= np.uint16)

class UnknownMon():
    def __init__(self):

        self.info = create_pkm_array()


class Pokebot_Gen1(Player):
    def __init__(self, account_configuration = None, *, avatar = None, battle_format = "gen9randombattle", log_level = None, max_concurrent_battles = 1, accept_open_team_sheet = False, save_replays = False, server_configuration = None, start_timer_on_battle_start = False, start_listening = True, ping_interval = 20, ping_timeout = 20, team = None):
        super().__init__(account_configuration, avatar=avatar, battle_format=battle_format, log_level=log_level, max_concurrent_battles=max_concurrent_battles, accept_open_team_sheet=accept_open_team_sheet, save_replays=save_replays, server_configuration=server_configuration, start_timer_on_battle_start=start_timer_on_battle_start, start_listening=start_listening, ping_interval=ping_interval, ping_timeout=ping_timeout, team=team)
        self.seed_dict = {}
        self.player_num = None
        self.opponent_num = None
        self.prev_embed_battle = None
        self.player_active_index = -1
        self.opponent_active_index = -1
        '''
        We represent this with 2 dimensional array. 12 is number of pokemon
        Each array contains required information
        We represent it as [Pokemon Species (dex number), Opponent or Player Pokemon(0 represents player and 1 represents opponent), Currently Active(0 is false, 1 is true), 
        Current Status Condition (Note:sleep turns are included here),Additional Status Conditions (Confusion/Leech Seed), Partial Trapped Turns, Toxic Counter,
          Current Health, Max Health, Base Speed, Current Speed, Current Special, Base Attack, Current Attack, Current Defense, Level, Speed boosts, Special boosts, 
          Attack boosts, Defense boosts, Accuracy boosts, Reflect-LightScreen, Move 1 Name, Move 1 PP, Move 2 Name, Move 2 PP, 
          Move 3 Name, Move 3 PP, Move 4 Name, Move 4 PP,prev_dmg (used for counter),substitute_hp (0 if no substitute is in place)]
        '''
        high_values = np.array([255,1,1,15,15,15,31,1023,1023,1023,1023,1023,1023,1023,1023,127,15,15,15,15,15,3,255,127,255,127,255,127,255,127,1023,511])
        high_matrix = np.tile(high_values, (12, 1)) 

        self.observation_space = Box(
                    low=0,  
                    high=high_matrix,  
                    shape=(12, 32), 
                    dtype=np.uint16
                )
        tea = self._team
        order_list = []
        if tea is None:
            self.teamnames = []
        else:
            team_names = []
            team_list = tea.team
            move_list = []
            for t in team_list:
                t = (str(t)).strip()
                mon_string = ''
                for c in t:
                    if c == ' ':
                        mon_string = mon_string.lower()
                        team_names.append(mon_string)
                        mon_num = dex_nums[mon_string]
                        order_list.append(mon_num)
                        break
                    else:
                        mon_string+=c
                count = 0
                moves = []
                move_string = ''
                for c in t:
                    if count == 4:
                        if c == ',' or c == '|':
                            m = Move(move_string,1)
                            moves.append(m)
                            move_string = ''
                        else:
                            move_string+=c
                    elif count == 5:
                        break
                    if c == '|':
                        count+=1
                move_list.append(moves)    
            self.original_mon_order = order_list[:]
            move_dict = dict(zip(team_names,move_list)) 
            self.teaminfo = move_dict       
            self.teamnames = team_names
            
    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, victory_value=1
        )

    # def get_state(self, battle: AbstractBattle):
    #     player_team = battle.team
    #     opponent_team = battle.opponent_team
    #     if battle.turn == 1:
    #         for mon in player_team:
    #             mon_tuple = (0,dex_nums[mon.species])
    #             self.seed_dict[mon_tuple] = 1
    #         for mon in opponent_team:
    #             mon_tuple = (1,dex_nums[mon.species])
    #             self.seed_dict[mon_tuple] = 1
    #     player_mon = battle.active_pokemon
    #     opponent_mon = battle.opponent_active_pokemon
    #     return (player_team,player_mon,opponent_team,opponent_mon)

    def embed_battle(self,battle:AbstractBattle):

        ### partial trapping, toxic, confusion, leech seed, counter substitute are going to be implemented later
        player_array_list = []
        opp_array_list = []
        if battle.turn == 1:

            player_team = list(battle.team.values())
            opponent_team = list(battle.opponent_team.values())
            if len(player_team)<6:
                pteam_names = []
                for mon in player_team:
                    pteam_names.append(mon.species)
                full_player_team = player_team + [item for item in self.teamnames if item not in pteam_names]
                zipped = zip_longest(full_player_team,opponent_team,fillvalue=UnknownMon())
            else:
                zipped = zip_longest(player_team,opponent_team,fillvalue=UnknownMon())
            partial_trap = 0
            toxic_counter = 0
            alt_status = 0
            sub_hp = 0
            index = 0
            

            for pmon,omon in zipped:
                if isinstance(omon,UnknownMon):
                    omon_array = omon.info
                else:
                    ospecies = omon.species
                    odex_num = dex_nums[ospecies]
                    o_act_base_speed = stats.loc[ospecies]['Speed_Total']
                    o_act_current_speed = o_act_base_speed
                    o_act_current_special = stats.loc[ospecies]['Special_Total']
                    o_act_base_attack = stats.loc[ospecies]['Attack_Total']
                    o_act_current_attack = o_act_base_attack
                    o_act_current_defense = stats.loc[ospecies]['Defense_Total']
                    omon_array = create_pkm_array(spcs = odex_num, player=1,active=1,curr_status=0,add_status=0,partial_trap=0,
                                                  toxic_counter=1,curr_health=omon.current_hp,max_health=omon.max_hp,base_speed=o_act_base_speed,curr_speed=o_act_current_speed,
                                                  curr_special=o_act_current_special,base_attack=o_act_base_attack,curr_attack=o_act_current_attack,curr_defense=o_act_current_defense,
                                                  level=100,speed_boosts = 7, special_boosts = 7, attack_boosts = 7, defense_boosts = 7, accuracy_boosts = 7,
                                                reflghtscreen = 0, mv1n = 1, mv1pp = 100, mv2n = 1, mv2pp = 100, mv3n = 1, mv3pp = 100, mv4n = 1,mv4pp = 100,prev_dmg=0,substitute_hp = 0)

                opp_array_list.append(omon_array)
                if isinstance(pmon,Pokemon):
                    pspecies = pmon.species
                    p_stats = pmon.stats
                    if p_stats['hp'] is None:
                        p_stats = {'hp': stats.loc[pspecies]['HP_Total'], 'atk':stats.loc[pspecies]['Attack_Total'] , 'def': stats.loc[pspecies]['Defense_Total'], 'spa': stats.loc[pspecies]['Special_Total'], 
                                'spd': stats.loc[pspecies]['Special_Total'], 'spe': stats.loc[pspecies]['Speed_Total']}
                    if pmon.active:
                        pactive_num = 1
                    else:
                        pactive_num = 0
                    
                    pdex_num = dex_nums[pspecies]
                    pmoves = pmon.moves
                    move_names = []
                    move_pp = []
                    for k,v in pmoves.items():
                        move_names.append(move_dict[k])
                        move_pp.append(v.current_pp)
                    if len(move_names)<4:
                        pad_list_len = 4-len(move_names)
                        pad_list = [0]*pad_list_len
                        move_names.extend(pad_list)
                        move_pp.extend(pad_list)
                    pmon_array = create_pkm_array(spcs=pdex_num,player=0,active=pactive_num,curr_status=0,add_status=0,partial_trap=0,
                                                toxic_counter=1,curr_health=p_stats['hp'],max_health=p_stats['hp'],base_speed=p_stats['spe'],curr_speed=p_stats['spe'],
                                                curr_special=p_stats['spd'],base_attack=p_stats['atk'],curr_attack=p_stats['atk'],curr_defense=p_stats['def'],
                                                level=100,speed_boosts=7,special_boosts=7,attack_boosts=7,defense_boosts=7,accuracy_boosts=7,reflghtscreen=0,
                                                mv1n=move_names[0],mv1pp=move_pp[0],mv2n=move_names[1],mv2pp=move_pp[1],mv3n=move_names[2],mv3pp=move_pp[2],
                                                mv4n=move_names[3],mv4pp=move_pp[3],prev_dmg=0,substitute_hp=0
                                                )
                    player_array_list.append(pmon_array)
                else:
                    pspecies = pmon
                    p_stats = {'hp': stats.loc[pspecies]['HP_Total'], 'atk':stats.loc[pspecies]['Attack_Total'] , 'def': stats.loc[pspecies]['Defense_Total'], 'spa': stats.loc[pspecies]['Special_Total'], 
                                'spd': stats.loc[pspecies]['Special_Total'], 'spe': stats.loc[pspecies]['Speed_Total']}
                    p_active_num = 0                    
                    pdex_num = dex_nums[pspecies]
                    pmoves = self.teaminfo[pspecies]
                    for mve in pmoves:
                        move_names.append(move_dict[mve.id])
                        move_pp.append(mve.current_pp)
                    if len(move_names)<4:
                        pad_list_len = 4-len(move_names)
                        pad_list = [0]*pad_list_len
                        move_names.extend(pad_list)
                        move_pp.extend(pad_list)
                    pmon_array = create_pkm_array(spcs=pdex_num,player=0,active=pactive_num,curr_status=0,add_status=0,partial_trap=0,
                                                toxic_counter=1,curr_health=p_stats['hp'],max_health=p_stats['hp'],base_speed=p_stats['spe'],curr_speed=p_stats['spe'],
                                                curr_special=p_stats['spd'],base_attack=p_stats['atk'],curr_attack=p_stats['atk'],curr_defense=p_stats['def'],
                                                level=100,speed_boosts=7,special_boosts=7,attack_boosts=7,defense_boosts=7,accuracy_boosts=7,reflghtscreen=0,
                                                mv1n=move_names[0],mv1pp=move_pp[0],mv2n=move_names[1],mv2pp=move_pp[1],mv3n=move_names[2],mv3pp=move_pp[2],
                                                mv4n=move_names[3],mv4pp=move_pp[3],prev_dmg=0,substitute_hp=0
                                                )
                    player_array_list.append(pmon_array)                    
            player_array = np.vstack(player_array_list)
            opp_array = np.vstack(opp_array_list)
            full_array = np.vstack((player_array,opp_array))
            self.prev_embed_battle = full_array.copy()
            return full_array
        player_array_list = []
        opp_array_list = []
        player_team = list(battle.team.values())
        opponent_team = list(battle.opponent_team.values())
        partial_trap_num = 0
        toxic_num = 1
        alt_status = 0
        sub_hp = 0 
        player_mon = battle.active_pokemon
        opponent_mon = battle.opponent_active_pokemon
        if opponent_mon.status is None:
            opp_status = 0
        else:
            opp_status = opponent_mon.status.value
        opp_para_flag = False
        opp_brn_flag = False
        play_para_flag = False
        play_swap_flag = False
        play_fnt_flag = False
        opp_swap_flag = False
        opp_fnt_flag = False
        play_brn_flag = False
        if opp_status == 1:
            opp_brn_flag = True
        elif opp_status == 4:
            opp_para_flag = True
        if player_mon.status is None:
            player_status = 0
        else:
            player_status = player_mon.status.value
        
        if player_status == 1:
            play_brn_flag = True
        elif player_status == 4:
            play_para_flag = True
        
        player_array = np.array([])
        opp_array = np.array([])
        last_ob_index = max(battle.observations.keys())
        observations = battle.observations[last_ob_index]
        
        if not self.player_num:
            for ev in observations.events:
                if str(self.username) in ev and 'player' in ev:
                    if 'p1' in ev or 'p1a' in ev or 'p1b' in ev:
                        self.player_num = 1
                        self.opponent_num = 2
                    else:
                        self.player_num = 2
                        self.opponent_num = 1
                    break
        events = observations.events
        index = 0
        if len(player_team)<6:
            pteam_names = []
            for mon in player_team:
                pteam_names.append(mon.species)
            full_player_team = player_team + [item for item in self.teamnames if item not in pteam_names]
        
            zipped = zip_longest(full_player_team,opponent_team,fillvalue=UnknownMon())
        else:
            zipped = zip_longest(player_team,opponent_team,fillvalue=UnknownMon())
        for pmon,omon in zipped:

            if isinstance(pmon,Pokemon):
                pspecies = pmon.species

                pdex_num = dex_nums[pspecies]

                pteam_num = 0

                p_stats = pmon.stats
                if p_stats['hp'] is None:
                    p_stats = {'hp': stats.loc[pspecies]['HP_Total'], 'atk':stats.loc[pspecies]['Attack_Total'] , 'def': stats.loc[pspecies]['Defense_Total'], 'spa': stats.loc[pspecies]['Special_Total'], 
                            'spd': stats.loc[pspecies]['Special_Total'], 'spe': stats.loc[pspecies]['Speed_Total']}
                mover = None
                if pmon.active:
                    pactive_num = 1
                    p_act_dex_num = pdex_num
                    p_act_base_speed = p_stats['spe']
                    p_act_base_attack = p_stats['atk']
                    p_act_current_special = p_stats['spd']
                    p_act_current_defense = p_stats['def']
                    p_boost_dict = pmon.boosts
                    p_act_accuracy_boosts = 7 + p_boost_dict['accuracy']
                    p_act_speed_boosts = 7 + p_boost_dict['spe']
                    p_act_attack_boosts = 7 + p_boost_dict['atk']
                    p_act_special_boosts = 7 + p_boost_dict['spd']
                    p_act_defense_boosts = 7 + p_boost_dict['def']
                    if pmon.status is None:
                        p_act_stat_condition = 0
                    else:
                        p_act_stat_condition = pmon.status.value
                    if p_act_stat_condition == 7:
                        p_act_stat_condition = 13
                    elif p_act_stat_condition == 6:
                        p_act_stat_condition+= (pmon.status_counter-1) # need to verify what they count as turn 0
                    p_act_hp_stat = pmon.current_hp
                    p_act_max_hp = pmon.max_hp
                    p_act_moves = pmon.moves
                    p_act_move_names = []
                    p_act_move_pp = []
                    for k,v in p_act_moves.items():
                        p_act_move_names.append(move_dict[k])
                        p_act_move_pp.append(v.current_pp)
                    if len(p_act_move_names)<4:
                        pad_list_len = 4-len(p_act_move_names)
                        pad_list = [0]*pad_list_len
                        p_act_move_names.extend(pad_list)
                        p_act_move_pp.extend(pad_list)
                    if index != self.player_active_index:
                        play_swap_flag = True
                        self.player_active_index = index

                    p_act_current_speed = self.prev_embed_battle[index][10]
                    if play_para_flag and play_swap_flag:
                        p_act_current_speed = max(1,p_act_current_speed//4)

                    
                    
                    p_act_current_attack = self.prev_embed_battle[index][13]
                    if play_brn_flag and play_swap_flag:
                        p_act_current_attack = max(1,p_act_current_attack//2)

                else:
                    pactive_num = 0

                if pactive_num == 0:
                    p_speed_boosts = 7
                    p_special_boosts = 7
                    p_attack_boosts = 7
                    p_defense_boosts = 7
                    p_accuracy_boosts = 7
                    p_base_speed = p_stats['spe'] 
                    p_current_speed = p_base_speed
                    p_base_attack = p_stats['atk']
                    p_current_attack = p_base_attack
                    p_current_special = p_stats['spd']
                    p_current_defense = p_stats['def']
                    if pmon.status is None:
                        pstat_condition = 0
                    else:
                        pstat_condition = pmon.status.value
                    if pstat_condition == 7:
                        pstat_condition = 13
                    elif pstat_condition == 6:
                        pstat_condition+= (pmon.status_counter-1) # need to verify what they count as turn 0
                    
                    php_stat = pmon.current_hp
                    p_max_hp = pmon.max_hp
                    pmoves = pmon.moves
                    move_names = []
                    move_pp = []
                    for k,v in pmoves.items():
                        move_names.append(move_dict[k])
                        move_pp.append(v.current_pp)
                    if len(move_names)<4:
                        pad_list_len = 4-len(move_names)
                        pad_list = [0]*pad_list_len
                        move_names.extend(pad_list)
                        move_pp.extend(pad_list)
                    p_mon_array = create_pkm_array(spcs = pdex_num,player=0,active=0,curr_status=pstat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                                                toxic_counter=toxic_num,curr_health=php_stat,max_health=p_max_hp,base_speed=p_base_speed,curr_speed=p_current_speed,
                                                curr_special=p_current_special,base_attack=p_base_attack,curr_attack=p_current_attack,curr_defense=p_current_defense,level=100,
                                                speed_boosts=p_speed_boosts,special_boosts=p_special_boosts,attack_boosts=p_attack_boosts,defense_boosts=p_defense_boosts,accuracy_boosts=p_accuracy_boosts,
                                                reflghtscreen=0,mv1n=move_names[0],mv1pp=move_pp[0],mv2n=move_names[1],mv2pp=move_pp[1],mv3n=move_names[2],mv3pp=move_pp[2],
                                                mv4n=move_names[3],mv4pp=move_pp[3],prev_dmg=0,substitute_hp=0
                                                ).reshape(1,-1)

                    
                    player_array_list.append(p_mon_array)
            else:
                pspecies = pmon
                p_stats = {'hp': stats.loc[pspecies]['HP_Total'], 'atk':stats.loc[pspecies]['Attack_Total'] , 'def': stats.loc[pspecies]['Defense_Total'], 'spa': stats.loc[pspecies]['Special_Total'], 
                            'spd': stats.loc[pspecies]['Special_Total'], 'spe': stats.loc[pspecies]['Speed_Total']}
                p_active_num = 0                    
                pdex_num = dex_nums[pspecies]
                pmoves = self.teaminfo[pspecies]
                for mve in pmoves:
                    move_names.append(move_dict[mve.id])
                    move_pp.append(mve.current_pp)
                if len(move_names)<4:
                    pad_list_len = 4-len(move_names)
                    pad_list = [0]*pad_list_len
                    move_names.extend(pad_list)
                    move_pp.extend(pad_list)
                p_mon_array = create_pkm_array(spcs=pdex_num,player=0,active=pactive_num,curr_status=0,add_status=0,partial_trap=0,
                                            toxic_counter=1,curr_health=p_stats['hp'],max_health=p_stats['hp'],base_speed=p_stats['spe'],curr_speed=p_stats['spe'],
                                            curr_special=p_stats['spd'],base_attack=p_stats['atk'],curr_attack=p_stats['atk'],curr_defense=p_stats['def'],
                                            level=100,speed_boosts=7,special_boosts=7,attack_boosts=7,defense_boosts=7,accuracy_boosts=7,reflghtscreen=0,
                                            mv1n=move_names[0],mv1pp=move_pp[0],mv2n=move_names[1],mv2pp=move_pp[1],mv3n=move_names[2],mv3pp=move_pp[2],
                                            mv4n=move_names[3],mv4pp=move_pp[3],prev_dmg=0,substitute_hp=0
                                            ).reshape(1,-1)
                player_array_list.append(p_mon_array)

            if isinstance(omon,UnknownMon):
                o_mon_array = omon.info
                o_mon_array = o_mon_array.reshape(1,-1)
                opp_array_list.append(o_mon_array) 
            else:
                ospecies = omon.species
                odex_num = dex_nums[ospecies]
                oteam_num = 1
                if omon.active:
                    oactive_num = 1
                    o_act_dex_num = odex_num
                    o_boost_dict = omon.boosts
                    o_act_accuracy_boosts = 7 + o_boost_dict['accuracy']
                    o_act_speed_boosts = 7 + o_boost_dict['spe']
                    o_act_attack_boosts = 7 + o_boost_dict['atk']
                    o_act_special_boosts = 7 + o_boost_dict['spd']
                    o_act_defense_boosts = 7 + o_boost_dict['def']
                    if omon.status is None:
                        o_act_stat_condition = 0
                    else:
                        o_act_stat_condition = omon.status.value
                    if o_act_stat_condition == 7:
                        o_act_stat_condition = 13
                    elif o_act_stat_condition == 6:
                        o_act_stat_condition+= (omon.status_counter-1) # need to verify what they count as turn 0
                    o_act_hp_stat = omon.current_hp
                    o_act_max_hp = omon.max_hp
                    o_act_moves = omon.moves
                    o_act_move_names = []
                    o_act_move_pp = []
                    for k,v in o_act_moves.items():
                        o_act_move_names.append(move_dict[k])
                        o_act_move_pp.append(v.current_pp)
                    if len(o_act_move_names)<4:
                        pad_list_len = 4-len(o_act_move_names)
                        pad_list = [0]*pad_list_len
                        o_act_move_names.extend(pad_list)
                        o_act_move_pp.extend(pad_list)
                    if index != self.opponent_active_index:
                        opp_swap_flag = True
                        self.opponent_active_index = index
                    if self.prev_embed_battle[index+6][9] == 0:
                        o_act_base_speed = stats.loc[ospecies]['Speed_Total']
                        o_act_current_speed = o_act_base_speed*boosts_dict[o_act_speed_boosts]
                        if opp_para_flag:
                            o_act_current_speed = max(1,o_act_current_speed//4)
                            
                    else:
                        o_act_base_speed = self.prev_embed_battle[index+6][9]
                        o_act_current_speed = self.prev_embed_battle[index+6][10]
                    if self.prev_embed_battle[index+6][11] == 0:
                        o_act_current_special = stats.loc[ospecies]['Special_Total']
                    else:
                        o_act_current_special = self.prev_embed_battle[index+6][11]
                    if self.prev_embed_battle[index+6][12] == 0:
                        o_act_base_attack = stats.loc[ospecies]['Attack_Total']
                        o_act_current_attack = o_act_base_attack
                    else:
                        o_act_base_attack = self.prev_embed_battle[index+6][12]
                        o_act_current_attack = self.prev_embed_battle[index+6][13]                      
                    if self.prev_embed_battle[index+6][14] == 0:
                        o_act_current_defense = stats.loc[ospecies]['Defense_Total']
                    else:
                        o_act_current_defense = self.prev_embed_battle[index+6][14]




                else:
                    oactive_num = 0

                if oactive_num == 0:
                    o_speed_boosts = 7
                    o_special_boosts = 7
                    o_attack_boosts = 7
                    o_defense_boosts = 7
                    o_accuracy_boosts = 7
                    o_base_speed = self.prev_embed_battle[index+6][9]
                    o_current_speed = o_base_speed
                    o_base_attack = self.prev_embed_battle[index+6][12]
                    o_current_attack = o_base_attack
                    o_current_special = self.prev_embed_battle[index+6][11]
                    o_current_defense = self.prev_embed_battle[index+6][14]
                    if omon.status is None:
                        ostat_condition = 0
                    else:
                        ostat_condition = omon.status.value
                    if ostat_condition == 7:
                        ostat_condition = 13
                    elif ostat_condition == 6:
                        ostat_condition+= (omon.status_counter-1) # need to verify what they count as turn 0
                    
                    ohp_stat = omon.current_hp
                    o_max_hp = omon.max_hp
                    omoves = omon.moves
                    move_names = []
                    move_pp = []
                    for k,v in omoves.items():
                        move_names.append(move_dict[k])
                        move_pp.append(v.current_pp)
                    if len(move_names)<4:
                        pad_list_len = 4-len(move_names)
                        pad_list = [1]*pad_list_len
                        pad_list_pp = [100]*pad_list_len
                        move_names.extend(pad_list)
                        move_pp.extend(pad_list_pp)
                    o_mon_array = create_pkm_array(spcs=odex_num,player=1,active=0,curr_status=ostat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                                                toxic_counter=toxic_num,curr_health=ohp_stat,max_health=o_max_hp,base_speed=o_base_speed,curr_speed=o_current_speed,
                                                curr_special=o_current_special,base_attack=o_base_attack,curr_attack=o_current_attack,curr_defense=o_current_defense,level=100,
                                                speed_boosts=o_speed_boosts,special_boosts=o_special_boosts,attack_boosts=o_attack_boosts,defense_boosts=o_defense_boosts,accuracy_boosts=o_accuracy_boosts,
                                                reflghtscreen=0,mv1n=move_names[0],mv1pp=move_pp[0],mv2n=move_names[1],mv2pp=move_pp[1],mv3n=move_names[2],mv3pp=move_pp[2],
                                                mv4n=move_names[3],mv4pp=move_pp[3],prev_dmg=0,substitute_hp=0
                                                ).reshape(1,-1)
            
                    opp_array_list.append(o_mon_array)      
            index+=1
        

        p_act_array = np.array([])
        o_act_array = np.array([])

        for e in events:
            event_name = e[1]
            if event_name == 'faint':
                fainter = e[2][1]
                if fainter == self.opponent_num:
                    o_act_array = create_pkm_array(spcs=o_act_dex_num,player=1,active=1,curr_status=o_act_stat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                                               toxic_counter=toxic_num,curr_health=o_act_hp_stat,max_health=o_act_max_hp,base_speed=o_act_base_speed,curr_speed=o_act_current_speed,
                                               curr_special=o_act_current_special,base_attack=o_act_base_attack,curr_attack=o_act_current_attack,curr_defense=o_act_current_defense,level=100,
                                               speed_boosts=o_act_speed_boosts,special_boosts=o_act_special_boosts,attack_boosts=o_act_attack_boosts,defense_boosts=o_act_defense_boosts,accuracy_boosts=o_act_accuracy_boosts,
                                               reflghtscreen=0,mv1n=o_act_move_names[0],mv1pp=o_act_move_pp[0],mv2n=o_act_move_names[1],mv2pp=o_act_move_pp[1],mv3n=o_act_move_names[2],mv3pp=o_act_move_pp[2],
                                              mv4n=o_act_move_names[3],mv4pp=o_act_move_pp[3],prev_dmg=0,substitute_hp=0).reshape(1,-1)
                else:
                    p_act_array = create_pkm_array(spcs=p_act_dex_num,player=1,active=1,curr_status=p_act_stat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                                               toxic_counter=toxic_num,curr_health=p_act_hp_stat,max_health=p_act_max_hp,base_speed=p_act_base_speed,curr_speed=p_act_current_speed,
                                               curr_special=p_act_current_special,base_attack=p_act_base_attack,curr_attack=p_act_current_attack,curr_defense=p_act_current_defense,level=100,
                                               speed_boosts=p_act_speed_boosts,special_boosts=p_act_special_boosts,attack_boosts=p_act_attack_boosts,defense_boosts=p_act_defense_boosts,accuracy_boosts=p_act_accuracy_boosts,
                                               reflghtscreen=0,mv1n=p_act_move_names[0],mv1pp=p_act_move_pp[0],mv2n=p_act_move_names[1],mv2pp=p_act_move_pp[1],mv3n=p_act_move_names[2],mv3pp=p_act_move_pp[2],
                                              mv4n=p_act_move_names[3],mv4pp=p_act_move_pp[3],prev_dmg=0,substitute_hp=0).reshape(1,-1)
            if event_name == 'move':
                mover = e[2][1]
                if mover == 1:
                    non_mover = 2
                else:
                    non_mover = 1
            if event_name == 'boost' or event_name == '-unboost':
                target = int(e[2][1])
                if mover == self.opponent_num and target == self.player_num:
                    if 'spe' in e:
                        p_act_current_speed = p_act_base_speed*boosts_dict[p_act_speed_boosts]
                        if play_para_flag:
                            p_act_current_speed = max(1,p_act_current_speed//4)
                    elif 'atk' in e:
                        p_act_current_attack = p_act_base_attack*boosts_dict[p_act_attack_boosts]
                        if play_brn_flag:
                            p_act_current_attack = max(1,p_act_current_attack//2)
                elif mover == self.opponent_num and target == mover:
                    if 'spe' in e:
                        o_act_current_speed = o_act_base_speed*boosts_dict[o_act_speed_boosts]
                        if play_para_flag:
                            p_act_current_speed = max(1,p_act_current_speed//4)
                    elif 'atk' in e:
                        o_act_current_attack = o_base_attack*boosts_dict[o_act_attack_boosts]
                        if play_brn_flag:
                            p_act_current_attack = max(1,p_act_current_attack//2)
                elif mover == self.player_num and target == self.opponent_num:
                    if 'spe' in e:
                        o_act_current_speed = o_act_base_speed*boosts_dict[o_act_speed_boosts]
                        if opp_para_flag:
                            o_act_current_speed = max(1,o_act_current_speed//4)
                    elif 'atk' in e:
                        o_act_current_attack = o_act_base_attack*boosts_dict[o_act_attack_boosts]
                        if opp_brn_flag:
                            o_act_current_attack = max(1,o_act_current_attack//2)
                elif mover == self.player_num and target == mover:
                    if 'spe' in e:
                        p_act_current_speed = p_act_base_speed*boosts_dict[p_act_speed_boosts]
                        if opp_para_flag:
                            o_act_current_speed = max(1,o_act_current_speed//4)
                    elif 'atk' in e:
                        p_act_current_attack = p_act_base_attack*boosts_dict[o_act_attack_boosts]
                        if opp_brn_flag:
                            o_act_current_attack = max(1,o_act_current_attack//2)
        if o_act_array.size == 0:
            o_act_array = create_pkm_array(spcs=o_act_dex_num,player=1,active=1,curr_status=o_act_stat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                            toxic_counter=toxic_num,curr_health=o_act_hp_stat,max_health=o_act_max_hp,base_speed=o_act_base_speed,curr_speed=o_act_current_speed,
                            curr_special=o_act_current_special,base_attack=o_act_base_attack,curr_attack=o_act_current_attack,curr_defense=o_act_current_defense,level=100,
                            speed_boosts=o_act_speed_boosts,special_boosts=o_act_special_boosts,attack_boosts=o_act_attack_boosts,defense_boosts=o_act_defense_boosts,accuracy_boosts=o_act_accuracy_boosts,
                            reflghtscreen=0,mv1n=o_act_move_names[0],mv1pp=o_act_move_pp[0],mv2n=o_act_move_names[1],mv2pp=o_act_move_pp[1],mv3n=o_act_move_names[2],mv3pp=o_act_move_pp[2],
                            mv4n=o_act_move_names[3],mv4pp=o_act_move_pp[3],substitute_hp=0).reshape(1,-1)
        if p_act_array.size == 0:
            p_act_array = create_pkm_array(spcs=p_act_dex_num,player=1,active=1,curr_status=p_act_stat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                                        toxic_counter=toxic_num,curr_health=p_act_hp_stat,max_health=p_act_max_hp,base_speed=p_act_base_speed,curr_speed=p_act_current_speed,
                                        curr_special=p_act_current_special,base_attack=p_act_base_attack,curr_attack=p_act_current_attack,curr_defense=p_act_current_defense,level=100,
                                        speed_boosts=p_act_speed_boosts,special_boosts=p_act_special_boosts,attack_boosts=p_act_attack_boosts,defense_boosts=p_act_defense_boosts,accuracy_boosts=p_act_accuracy_boosts,
                                        reflghtscreen=0,mv1n=p_act_move_names[0],mv1pp=p_act_move_pp[0],mv2n=p_act_move_names[1],mv2pp=p_act_move_pp[1],mv3n=p_act_move_names[2],mv3pp=p_act_move_pp[2],
                                        mv4n=p_act_move_names[3],mv4pp=p_act_move_pp[3],substitute_hp=0).reshape(1,-1)
        player_array_list.insert(self.player_active_index,p_act_array)

        vec_dict = {vec[0][0]: vec for vec in player_array_list}
        
        ordered_player_array_list = [vec_dict[num] for num in self.original_mon_order]
        player_array = np.vstack(ordered_player_array_list)
        opp_array = np.vstack(opp_array_list)
        # player_array = np.insert(player_array,self.player_active_index,p_act_array,axis=0)
        opp_array = np.insert(opp_array,self.opponent_active_index,o_act_array,axis=0)

        full_array = np.concatenate((player_array,opp_array),axis=0)
        self.prev_embed_battle = full_array.copy()

    def choose_move(self, battle):
        embed = self.embed_battle(battle)
        return self.choose_random_move(battle) 





class _EnvPlayer_Mod(_EnvPlayer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.seed_dict = {}
        self.player_num = None
        self.opponent_num = None
        self.prev_embed_battle = None
        self.player_active_index = -1
        self.opponent_active_index = -1
        self.full_team = None
        tea = self._team
        order_list = []
        if tea is None:
            self.teamnames = []
        else:
            team_names = []
            team_list = tea.team
            move_list = []

            for t in team_list:
                t = (str(t)).strip()
                mon_string = ''
                for c in t:
                    if c == ' ' or c == '|':
                        mon_string = mon_string.lower()
                        team_names.append(mon_string)
                        mon_num = dex_nums[mon_string]
                        order_list.append(mon_num)
                        break
                    else:
                        mon_string+=c
                count = 0
                moves = []
                move_string = ''
                for c in t:
                    if count == 4:
                        if c == ',' or c == '|':
                            m = Move(move_string,1)
                            moves.append(m)
                            move_string = ''
                        else:
                            move_string+=c
                    elif count == 5:
                        break
                    if c == '|':
                        count+=1
                move_list.append(moves)    
            self.original_mon_order = order_list[:]
            move_dict = dict(zip(team_names,move_list)) 
            self.teaminfo = move_dict       
            self.teamnames = team_names

    def get_full_team(self,battle):
        # to be called on turn 1
        current_mon = battle.active_pokemon
        other_mon = battle.available_switches
        fullteam = [current_mon]+other_mon
        self.full_team = fullteam.copy()

    def update_team(self, team):

        if isinstance(team, Teambuilder):
            self._team = team
        else:
            self._team = ConstantTeambuilder(team)
        
        tea = self._team
        order_list = []
        
        if tea is None:
            self.teamnames = []
        else:
            team_names = []
            team_list = tea.team

            
            move_list = []
            for t in team_list:
                t = (str(t)).strip()
                mon_string = ''
                for c in t:
                    if c == ' ' or c == '|':
                        mon_string = mon_string.lower()
                        team_names.append(mon_string)
                        mon_num = dex_nums[mon_string]
                        order_list.append(mon_num)
                        break
                    else:

                        mon_string+=c
                count = 0
                moves = []
                move_string = ''
                for c in t:
                    if count == 4:
                        if c == ',' or c == '|':
                            m = Move(move_string,1)
                            moves.append(m)
                            move_string = ''
                        else:
                            move_string+=c
                    elif count == 5:
                        break
                    if c == '|':
                        count+=1
                move_list.append(moves)    
            self.original_mon_order = order_list[:]

            move_dict = dict(zip(team_names,move_list)) 
            self.teaminfo = move_dict       
            self.teamnames = team_names        
    
class Pokebot_Gen1_Environment(PokeEnv):
    metadata = {'name':'pokemon_gen_1_v1'}


    def __init__(
        self,
        *,
        account_configuration1: Optional[AccountConfiguration] = None,
        account_configuration2: Optional[AccountConfiguration] = None,
        avatar: Optional[int] = None,
        battle_format: str = "gen1ou",
        log_level: Optional[int] = None,
        save_replays: Union[bool, str] = False,
        server_configuration: Optional[
            ServerConfiguration
        ] = LocalhostServerConfiguration,
        accept_open_team_sheet: Optional[bool] = False,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        open_timeout: Optional[float] = 10.0,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        team1: Optional[Union[str, Teambuilder]] = None,
        team2:Optional[Union[str, Teambuilder]] = None,
        start_challenging: bool = False,
        fake: bool = False,
        strict: bool = True,
    ):
        
        '''
        We represent this with 2 dimensional array. 12 is number of pokemon
        Each array contains required information
        We represent it as [Pokemon Species (dex number), Opponent or Player Pokemon(0 represents player and 1 represents opponent), Currently Active(0 is false, 1 is true), 
        Current Status Condition (Note:sleep turns are included here),Additional Status Conditions (Confusion/Leech Seed), Partial Trapped Turns, Toxic Counter,
          Current Health, Max Health, Base Speed, Current Speed, Current Special, Base Attack, Current Attack, Current Defense, Level, Speed boosts, Special boosts, 
          Attack boosts, Defense boosts, Accuracy boosts, Reflect-LightScreen, Move 1 Name, Move 1 PP, Move 2 Name, Move 2 PP, 
          Move 3 Name, Move 3 PP, Move 4 Name, Move 4 PP,prev_dmg (used for counter),substitute_hp (0 if no substitute is in place)]
        '''
        self._np_random: Optional[Generator] = None
        if team1 is None:
            team1 = team_generator_alt()
        if team2 is None:
            team2 = team_generator_alt
        unique_id = str(uuid.uuid4())[:8]
        self.agent_1_name = f"rl_agent_1_{unique_id}"
        self.agent_2_name = f"rl_agent_2_{unique_id}"
        self.agent1 = _EnvPlayer_Mod(
            username=self.agent_1_name,  # type: ignore
            account_configuration=account_configuration1,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            max_concurrent_battles=10,
            save_replays=save_replays,
            server_configuration=server_configuration,
            accept_open_team_sheet=accept_open_team_sheet,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            open_timeout=open_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            team=team1,
        )
        self.agent2 = _EnvPlayer_Mod(
            username=self.agent_2_name,  # type: ignore
            account_configuration=account_configuration2,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            max_concurrent_battles=10,
            save_replays=save_replays,
            server_configuration=server_configuration,
            accept_open_team_sheet=accept_open_team_sheet,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            open_timeout=open_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            team=team2,
        )

        self.agents: List[str] = []
        self.possible_agents = [self.agent1.username, self.agent2.username]
        self._action_spaces = {
            self.possible_agents[0]: Discrete(12),
            self.possible_agents[1]: Discrete(12)
        }
        high_values = np.array([255,1,1,15,15,15,31,1023,1023,1023,1023,1023,1023,1023,1023,127,15,15,15,15,15,3,255,127,255,127,255,127,255,127,1023,511])
        high_matrix = np.tile(high_values, (12, 1)) 


        self._observation_spaces = {
            self.possible_agents[0]: Box(
                    low=0,  
                    high=high_matrix,  
                    shape=(12, 32), 
                    dtype=np.uint16
                ),
            self.possible_agents[1]:Box(
                    low=0,  
                    high=high_matrix,  
                    shape=(12, 32), 
                    dtype=np.uint16
                )
        }

        self.battle1: Optional[AbstractBattle] = None
        self.battle2: Optional[AbstractBattle] = None
        self.agent1_to_move = False
        self.agent2_to_move = False
        self.fake = fake
        self.strict = strict

        self._reward_buffer: WeakKeyDictionary[AbstractBattle, float] = (
            WeakKeyDictionary()
        )
        self._keep_challenging: bool = False
        self._challenge_task = None
        if start_challenging:
            self._keep_challenging = True
            self._challenge_task = asyncio.run_coroutine_threadsafe(
                self._challenge_loop(), POKE_LOOP
            )
    def observation_space(self, agent): 
        return self._observation_spaces[agent]



    def action_space(self, agent):
        return self._action_spaces[agent]



    def embed_battle(self, battle):
    ### partial trapping, toxic, confusion, leech seed, counter substitute are going to be implemented later
        name = battle.player_username
        if name == self.agent1.username:
            active_agent = self.agent1
        else:
            active_agent = self.agent2
        player_array_list = []
        opp_array_list = []
        if battle.turn == 1:

            player_team = list(battle.team.values())
            opponent_team = list(battle.opponent_team.values())
            if len(player_team)<6:
                pteam_names = []
                for mon in player_team:
                    pteam_names.append(mon.species)
                full_player_team = player_team + [item for item in active_agent.teamnames if item not in pteam_names]
                zipped = zip_longest(full_player_team,opponent_team,fillvalue=UnknownMon())
            else:
                zipped = zip_longest(player_team,opponent_team,fillvalue=UnknownMon())
            partial_trap = 0
            toxic_counter = 0
            alt_status = 0
            sub_hp = 0
            index = 0
            

            for pmon,omon in zipped:
                if isinstance(omon,UnknownMon):
                    omon_array = omon.info
                else:
                    ospecies = omon.species
                    odex_num = dex_nums[ospecies]
                    o_act_base_speed = stats.loc[ospecies]['Speed_Total']
                    o_act_current_speed = o_act_base_speed
                    o_act_current_special = stats.loc[ospecies]['Special_Total']
                    o_act_base_attack = stats.loc[ospecies]['Attack_Total']
                    o_act_current_attack = o_act_base_attack
                    o_act_current_defense = stats.loc[ospecies]['Defense_Total']
                    omon_array = create_pkm_array(spcs = odex_num, player=1,active=1,curr_status=0,add_status=0,partial_trap=0,
                                                  toxic_counter=1,curr_health=omon.current_hp,max_health=omon.max_hp,base_speed=o_act_base_speed,curr_speed=o_act_current_speed,
                                                  curr_special=o_act_current_special,base_attack=o_act_base_attack,curr_attack=o_act_current_attack,curr_defense=o_act_current_defense,
                                                  level=100,speed_boosts = 7, special_boosts = 7, attack_boosts = 7, defense_boosts = 7, accuracy_boosts = 7,
                                                reflghtscreen = 0, mv1n = 1, mv1pp = 100, mv2n = 1, mv2pp = 100, mv3n = 1, mv3pp = 100, mv4n = 1,mv4pp = 100,prev_dmg=0,substitute_hp = 0)

                opp_array_list.append(omon_array)
                if isinstance(pmon,Pokemon):
                    pspecies = pmon.species
                    p_stats = pmon.stats
                    if p_stats['hp'] is None:
                        p_stats = {'hp': stats.loc[pspecies]['HP_Total'], 'atk':stats.loc[pspecies]['Attack_Total'] , 'def': stats.loc[pspecies]['Defense_Total'], 'spa': stats.loc[pspecies]['Special_Total'], 
                                'spd': stats.loc[pspecies]['Special_Total'], 'spe': stats.loc[pspecies]['Speed_Total']}
                    if pmon.active:
                        pactive_num = 1
                    else:
                        pactive_num = 0
                    
                    pdex_num = dex_nums[pspecies]
                    pmoves = pmon.moves
                    move_names = []
                    move_pp = []
                    for k,v in pmoves.items():
                        move_names.append(move_dict[k])
                        move_pp.append(v.current_pp)
                    if len(move_names)<4:
                        pad_list_len = 4-len(move_names)
                        pad_list = [0]*pad_list_len
                        move_names.extend(pad_list)
                        move_pp.extend(pad_list)
                    pmon_array = create_pkm_array(spcs=pdex_num,player=0,active=pactive_num,curr_status=0,add_status=0,partial_trap=0,
                                                toxic_counter=1,curr_health=p_stats['hp'],max_health=p_stats['hp'],base_speed=p_stats['spe'],curr_speed=p_stats['spe'],
                                                curr_special=p_stats['spd'],base_attack=p_stats['atk'],curr_attack=p_stats['atk'],curr_defense=p_stats['def'],
                                                level=100,speed_boosts=7,special_boosts=7,attack_boosts=7,defense_boosts=7,accuracy_boosts=7,reflghtscreen=0,
                                                mv1n=move_names[0],mv1pp=move_pp[0],mv2n=move_names[1],mv2pp=move_pp[1],mv3n=move_names[2],mv3pp=move_pp[2],
                                                mv4n=move_names[3],mv4pp=move_pp[3],prev_dmg=0,substitute_hp=0
                                                )
                    player_array_list.append(pmon_array)
                else:
                    pspecies = pmon
                    p_stats = {'hp': stats.loc[pspecies]['HP_Total'], 'atk':stats.loc[pspecies]['Attack_Total'] , 'def': stats.loc[pspecies]['Defense_Total'], 'spa': stats.loc[pspecies]['Special_Total'], 
                                'spd': stats.loc[pspecies]['Special_Total'], 'spe': stats.loc[pspecies]['Speed_Total']}
                    p_active_num = 0                    
                    pdex_num = dex_nums[pspecies]
                    pmoves = active_agent.teaminfo[pspecies]
                    for mve in pmoves:
                        move_names.append(move_dict[mve.id])
                        move_pp.append(mve.current_pp)
                    if len(move_names)<4:
                        pad_list_len = 4-len(move_names)
                        pad_list = [0]*pad_list_len
                        move_names.extend(pad_list)
                        move_pp.extend(pad_list)
                    pmon_array = create_pkm_array(spcs=pdex_num,player=0,active=pactive_num,curr_status=0,add_status=0,partial_trap=0,
                                                toxic_counter=1,curr_health=p_stats['hp'],max_health=p_stats['hp'],base_speed=p_stats['spe'],curr_speed=p_stats['spe'],
                                                curr_special=p_stats['spd'],base_attack=p_stats['atk'],curr_attack=p_stats['atk'],curr_defense=p_stats['def'],
                                                level=100,speed_boosts=7,special_boosts=7,attack_boosts=7,defense_boosts=7,accuracy_boosts=7,reflghtscreen=0,
                                                mv1n=move_names[0],mv1pp=move_pp[0],mv2n=move_names[1],mv2pp=move_pp[1],mv3n=move_names[2],mv3pp=move_pp[2],
                                                mv4n=move_names[3],mv4pp=move_pp[3],prev_dmg=0,substitute_hp=0
                                                )
                    player_array_list.append(pmon_array)                    
            player_array = np.vstack(player_array_list)
            opp_array = np.vstack(opp_array_list)
            full_array = np.vstack((player_array,opp_array))
            active_agent.prev_embed_battle = full_array.copy()
            return full_array
        player_array_list = []
        opp_array_list = []
        player_team = list(battle.team.values())
        opponent_team = list(battle.opponent_team.values())
        partial_trap_num = 0
        toxic_num = 1
        alt_status = 0
        sub_hp = 0 
        player_mon = battle.active_pokemon
        opponent_mon = battle.opponent_active_pokemon
        if opponent_mon.status is None:
            opp_status = 0
        else:
            opp_status = opponent_mon.status.value
        opp_para_flag = False
        opp_brn_flag = False
        play_para_flag = False
        play_swap_flag = False
        play_fnt_flag = False
        opp_swap_flag = False
        opp_fnt_flag = False
        play_brn_flag = False
        if opp_status == 1:
            opp_brn_flag = True
        elif opp_status == 4:
            opp_para_flag = True
        if player_mon.status is None:
            player_status = 0
        else:
            player_status = player_mon.status.value
        
        if player_status == 1:
            play_brn_flag = True
        elif player_status == 4:
            play_para_flag = True
        
        player_array = np.array([])
        opp_array = np.array([])
        last_ob_index = max(battle.observations.keys())
        observations = battle.observations[last_ob_index]
        
        if not active_agent.player_num:
            for ev in observations.events:
                if str(active_agent.username) in ev and 'player' in ev:
                    if 'p1' in ev or 'p1a' in ev or 'p1b' in ev:
                        active_agent.player_num = 1
                        active_agent.opponent_num = 2
                    else:
                        active_agent.player_num = 2
                        active_agent.opponent_num = 1
                    break
        events = observations.events
        index = 0
        if len(player_team)<6:
            pteam_names = []
            for mon in player_team:
                pteam_names.append(mon.species)
            full_player_team = player_team + [item for item in active_agent.teamnames if item not in pteam_names]
        
            zipped = zip_longest(full_player_team,opponent_team,fillvalue=UnknownMon())
        else:
            zipped = zip_longest(player_team,opponent_team,fillvalue=UnknownMon())
        for pmon,omon in zipped:

            if isinstance(pmon,Pokemon):
                pspecies = pmon.species

                pdex_num = dex_nums[pspecies]

                pteam_num = 0

                p_stats = pmon.stats
                if p_stats['hp'] is None:
                    p_stats = {'hp': stats.loc[pspecies]['HP_Total'], 'atk':stats.loc[pspecies]['Attack_Total'] , 'def': stats.loc[pspecies]['Defense_Total'], 'spa': stats.loc[pspecies]['Special_Total'], 
                            'spd': stats.loc[pspecies]['Special_Total'], 'spe': stats.loc[pspecies]['Speed_Total']}
                mover = None
                if pmon.active:
                    pactive_num = 1
                    p_act_dex_num = pdex_num
                    p_act_base_speed = p_stats['spe']
                    p_act_base_attack = p_stats['atk']
                    p_act_current_special = p_stats['spd']
                    p_act_current_defense = p_stats['def']
                    p_boost_dict = pmon.boosts
                    p_act_accuracy_boosts = 7 + p_boost_dict['accuracy']
                    p_act_speed_boosts = 7 + p_boost_dict['spe']
                    p_act_attack_boosts = 7 + p_boost_dict['atk']
                    p_act_special_boosts = 7 + p_boost_dict['spd']
                    p_act_defense_boosts = 7 + p_boost_dict['def']
                    if pmon.status is None:
                        p_act_stat_condition = 0
                    else:
                        p_act_stat_condition = pmon.status.value
                    if p_act_stat_condition == 7:
                        p_act_stat_condition = 13
                    elif p_act_stat_condition == 6:
                        p_act_stat_condition+= (pmon.status_counter-1) # need to verify what they count as turn 0
                    p_act_hp_stat = pmon.current_hp
                    p_act_max_hp = pmon.max_hp
                    p_act_moves = pmon.moves
                    p_act_move_names = []
                    p_act_move_pp = []
                    for k,v in p_act_moves.items():
                        p_act_move_names.append(move_dict[k])
                        p_act_move_pp.append(v.current_pp)
                    if len(p_act_move_names)<4:
                        pad_list_len = 4-len(p_act_move_names)
                        pad_list = [0]*pad_list_len
                        p_act_move_names.extend(pad_list)
                        p_act_move_pp.extend(pad_list)
                    if index != active_agent.player_active_index:
                        play_swap_flag = True
                        active_agent.player_active_index = index

                    p_act_current_speed = active_agent.prev_embed_battle[index][10]
                    if play_para_flag and play_swap_flag:
                        p_act_current_speed = max(1,p_act_current_speed//4)

                    
                    
                    p_act_current_attack = active_agent.prev_embed_battle[index][13]
                    if play_brn_flag and play_swap_flag:
                        p_act_current_attack = max(1,p_act_current_attack//2)

                else:
                    pactive_num = 0

                if pactive_num == 0:
                    p_speed_boosts = 7
                    p_special_boosts = 7
                    p_attack_boosts = 7
                    p_defense_boosts = 7
                    p_accuracy_boosts = 7
                    p_base_speed = p_stats['spe'] 
                    p_current_speed = p_base_speed
                    p_base_attack = p_stats['atk']
                    p_current_attack = p_base_attack
                    p_current_special = p_stats['spd']
                    p_current_defense = p_stats['def']
                    if pmon.status is None:
                        pstat_condition = 0
                    else:
                        pstat_condition = pmon.status.value
                    if pstat_condition == 7:
                        pstat_condition = 13
                    elif pstat_condition == 6:
                        pstat_condition+= (pmon.status_counter-1) # need to verify what they count as turn 0
                    
                    php_stat = pmon.current_hp
                    p_max_hp = pmon.max_hp
                    pmoves = pmon.moves
                    move_names = []
                    move_pp = []
                    for k,v in pmoves.items():
                        move_names.append(move_dict[k])
                        move_pp.append(v.current_pp)
                    if len(move_names)<4:
                        pad_list_len = 4-len(move_names)
                        pad_list = [0]*pad_list_len
                        move_names.extend(pad_list)
                        move_pp.extend(pad_list)
                    p_mon_array = create_pkm_array(spcs = pdex_num,player=0,active=0,curr_status=pstat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                                                toxic_counter=toxic_num,curr_health=php_stat,max_health=p_max_hp,base_speed=p_base_speed,curr_speed=p_current_speed,
                                                curr_special=p_current_special,base_attack=p_base_attack,curr_attack=p_current_attack,curr_defense=p_current_defense,level=100,
                                                speed_boosts=p_speed_boosts,special_boosts=p_special_boosts,attack_boosts=p_attack_boosts,defense_boosts=p_defense_boosts,accuracy_boosts=p_accuracy_boosts,
                                                reflghtscreen=0,mv1n=move_names[0],mv1pp=move_pp[0],mv2n=move_names[1],mv2pp=move_pp[1],mv3n=move_names[2],mv3pp=move_pp[2],
                                                mv4n=move_names[3],mv4pp=move_pp[3],prev_dmg=0,substitute_hp=0
                                                ).reshape(1,-1)

                    
                    player_array_list.append(p_mon_array)
            else:
                pspecies = pmon
                p_stats = {'hp': stats.loc[pspecies]['HP_Total'], 'atk':stats.loc[pspecies]['Attack_Total'] , 'def': stats.loc[pspecies]['Defense_Total'], 'spa': stats.loc[pspecies]['Special_Total'], 
                            'spd': stats.loc[pspecies]['Special_Total'], 'spe': stats.loc[pspecies]['Speed_Total']}
                p_active_num = 0                    
                pdex_num = dex_nums[pspecies]
                pmoves = active_agent.teaminfo[pspecies]
                for mve in pmoves:
                    move_names.append(move_dict[mve.id])
                    move_pp.append(mve.current_pp)
                if len(move_names)<4:
                    pad_list_len = 4-len(move_names)
                    pad_list = [0]*pad_list_len
                    move_names.extend(pad_list)
                    move_pp.extend(pad_list)
                p_mon_array = create_pkm_array(spcs=pdex_num,player=0,active=pactive_num,curr_status=0,add_status=0,partial_trap=0,
                                            toxic_counter=1,curr_health=p_stats['hp'],max_health=p_stats['hp'],base_speed=p_stats['spe'],curr_speed=p_stats['spe'],
                                            curr_special=p_stats['spd'],base_attack=p_stats['atk'],curr_attack=p_stats['atk'],curr_defense=p_stats['def'],
                                            level=100,speed_boosts=7,special_boosts=7,attack_boosts=7,defense_boosts=7,accuracy_boosts=7,reflghtscreen=0,
                                            mv1n=move_names[0],mv1pp=move_pp[0],mv2n=move_names[1],mv2pp=move_pp[1],mv3n=move_names[2],mv3pp=move_pp[2],
                                            mv4n=move_names[3],mv4pp=move_pp[3],prev_dmg=0,substitute_hp=0
                                            ).reshape(1,-1)
                player_array_list.append(p_mon_array)

            if isinstance(omon,UnknownMon):
                o_mon_array = omon.info
                o_mon_array = o_mon_array.reshape(1,-1)
                opp_array_list.append(o_mon_array) 
            else:
                ospecies = omon.species
                odex_num = dex_nums[ospecies]
                oteam_num = 1
                if omon.active:
                    oactive_num = 1
                    o_act_dex_num = odex_num
                    o_boost_dict = omon.boosts
                    o_act_accuracy_boosts = 7 + o_boost_dict['accuracy']
                    o_act_speed_boosts = 7 + o_boost_dict['spe']
                    o_act_attack_boosts = 7 + o_boost_dict['atk']
                    o_act_special_boosts = 7 + o_boost_dict['spd']
                    o_act_defense_boosts = 7 + o_boost_dict['def']
                    if omon.status is None:
                        o_act_stat_condition = 0
                    else:
                        o_act_stat_condition = omon.status.value
                    if o_act_stat_condition == 7:
                        o_act_stat_condition = 13
                    elif o_act_stat_condition == 6:
                        o_act_stat_condition+= (omon.status_counter-1) # need to verify what they count as turn 0
                    o_act_hp_stat = omon.current_hp
                    o_act_max_hp = omon.max_hp
                    o_act_moves = omon.moves
                    o_act_move_names = []
                    o_act_move_pp = []
                    for k,v in o_act_moves.items():
                        o_act_move_names.append(move_dict[k])
                        o_act_move_pp.append(v.current_pp)
                    if len(o_act_move_names)<4:
                        pad_list_len = 4-len(o_act_move_names)
                        pad_list = [0]*pad_list_len
                        o_act_move_names.extend(pad_list)
                        o_act_move_pp.extend(pad_list)
                    if index != active_agent.opponent_active_index:
                        opp_swap_flag = True
                        active_agent.opponent_active_index = index
                    if active_agent.prev_embed_battle[index+6][9] == 0:
                        o_act_base_speed = stats.loc[ospecies]['Speed_Total']
                        o_act_current_speed = o_act_base_speed*boosts_dict[o_act_speed_boosts]
                        if opp_para_flag:
                            o_act_current_speed = max(1,o_act_current_speed//4)
                            
                    else:
                        o_act_base_speed = active_agent.prev_embed_battle[index+6][9]
                        o_act_current_speed = active_agent.prev_embed_battle[index+6][10]
                    if active_agent.prev_embed_battle[index+6][11] == 0:
                        o_act_current_special = stats.loc[ospecies]['Special_Total']
                    else:
                        o_act_current_special = active_agent.prev_embed_battle[index+6][11]
                    if active_agent.prev_embed_battle[index+6][12] == 0:
                        o_act_base_attack = stats.loc[ospecies]['Attack_Total']
                        o_act_current_attack = o_act_base_attack
                    else:
                        o_act_base_attack = active_agent.prev_embed_battle[index+6][12]
                        o_act_current_attack = active_agent.prev_embed_battle[index+6][13]                      
                    if active_agent.prev_embed_battle[index+6][14] == 0:
                        o_act_current_defense = stats.loc[ospecies]['Defense_Total']
                    else:
                        o_act_current_defense = active_agent.prev_embed_battle[index+6][14]




                else:
                    oactive_num = 0

                if oactive_num == 0:
                    o_speed_boosts = 7
                    o_special_boosts = 7
                    o_attack_boosts = 7
                    o_defense_boosts = 7
                    o_accuracy_boosts = 7
                    o_base_speed = active_agent.prev_embed_battle[index+6][9]
                    o_current_speed = o_base_speed
                    o_base_attack = active_agent.prev_embed_battle[index+6][12]
                    o_current_attack = o_base_attack
                    o_current_special = active_agent.prev_embed_battle[index+6][11]
                    o_current_defense = active_agent.prev_embed_battle[index+6][14]
                    if omon.status is None:
                        ostat_condition = 0
                    else:
                        ostat_condition = omon.status.value
                    if ostat_condition == 7:
                        ostat_condition = 13
                    elif ostat_condition == 6:
                        ostat_condition+= (omon.status_counter-1) # need to verify what they count as turn 0
                    
                    ohp_stat = omon.current_hp
                    o_max_hp = omon.max_hp
                    omoves = omon.moves
                    move_names = []
                    move_pp = []
                    for k,v in omoves.items():
                        move_names.append(move_dict[k])
                        move_pp.append(v.current_pp)
                    if len(move_names)<4:
                        pad_list_len = 4-len(move_names)
                        pad_list = [1]*pad_list_len
                        pad_list_pp = [100]*pad_list_len
                        move_names.extend(pad_list)
                        move_pp.extend(pad_list_pp)
                    o_mon_array = create_pkm_array(spcs=odex_num,player=1,active=0,curr_status=ostat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                                                toxic_counter=toxic_num,curr_health=ohp_stat,max_health=o_max_hp,base_speed=o_base_speed,curr_speed=o_current_speed,
                                                curr_special=o_current_special,base_attack=o_base_attack,curr_attack=o_current_attack,curr_defense=o_current_defense,level=100,
                                                speed_boosts=o_speed_boosts,special_boosts=o_special_boosts,attack_boosts=o_attack_boosts,defense_boosts=o_defense_boosts,accuracy_boosts=o_accuracy_boosts,
                                                reflghtscreen=0,mv1n=move_names[0],mv1pp=move_pp[0],mv2n=move_names[1],mv2pp=move_pp[1],mv3n=move_names[2],mv3pp=move_pp[2],
                                                mv4n=move_names[3],mv4pp=move_pp[3],prev_dmg=0,substitute_hp=0
                                                ).reshape(1,-1)
            
                    opp_array_list.append(o_mon_array)      
            index+=1
        

        p_act_array = np.array([])
        o_act_array = np.array([])

        for e in events:
            event_name = e[1]
            if event_name == 'faint':
                fainter = e[2][1]
                if fainter == active_agent.opponent_num:
                    o_act_array = create_pkm_array(spcs=o_act_dex_num,player=1,active=1,curr_status=o_act_stat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                                               toxic_counter=toxic_num,curr_health=o_act_hp_stat,max_health=o_act_max_hp,base_speed=o_act_base_speed,curr_speed=o_act_current_speed,
                                               curr_special=o_act_current_special,base_attack=o_act_base_attack,curr_attack=o_act_current_attack,curr_defense=o_act_current_defense,level=100,
                                               speed_boosts=o_act_speed_boosts,special_boosts=o_act_special_boosts,attack_boosts=o_act_attack_boosts,defense_boosts=o_act_defense_boosts,accuracy_boosts=o_act_accuracy_boosts,
                                               reflghtscreen=0,mv1n=o_act_move_names[0],mv1pp=o_act_move_pp[0],mv2n=o_act_move_names[1],mv2pp=o_act_move_pp[1],mv3n=o_act_move_names[2],mv3pp=o_act_move_pp[2],
                                              mv4n=o_act_move_names[3],mv4pp=o_act_move_pp[3],prev_dmg=0,substitute_hp=0).reshape(1,-1)
                else:
                    p_act_array = create_pkm_array(spcs=p_act_dex_num,player=1,active=1,curr_status=p_act_stat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                                               toxic_counter=toxic_num,curr_health=p_act_hp_stat,max_health=p_act_max_hp,base_speed=p_act_base_speed,curr_speed=p_act_current_speed,
                                               curr_special=p_act_current_special,base_attack=p_act_base_attack,curr_attack=p_act_current_attack,curr_defense=p_act_current_defense,level=100,
                                               speed_boosts=p_act_speed_boosts,special_boosts=p_act_special_boosts,attack_boosts=p_act_attack_boosts,defense_boosts=p_act_defense_boosts,accuracy_boosts=p_act_accuracy_boosts,
                                               reflghtscreen=0,mv1n=p_act_move_names[0],mv1pp=p_act_move_pp[0],mv2n=p_act_move_names[1],mv2pp=p_act_move_pp[1],mv3n=p_act_move_names[2],mv3pp=p_act_move_pp[2],
                                              mv4n=p_act_move_names[3],mv4pp=p_act_move_pp[3],prev_dmg=0,substitute_hp=0).reshape(1,-1)
            if event_name == 'move':
                mover = e[2][1]
                if mover == 1:
                    non_mover = 2
                else:
                    non_mover = 1
            if event_name == 'boost' or event_name == '-unboost':
                target = int(e[2][1])
                if mover == active_agent.opponent_num and target == active_agent.player_num:
                    if 'spe' in e:
                        p_act_current_speed = p_act_base_speed*boosts_dict[p_act_speed_boosts]
                        if play_para_flag:
                            p_act_current_speed = max(1,p_act_current_speed//4)
                    elif 'atk' in e:
                        p_act_current_attack = p_act_base_attack*boosts_dict[p_act_attack_boosts]
                        if play_brn_flag:
                            p_act_current_attack = max(1,p_act_current_attack//2)
                elif mover == active_agent.opponent_num and target == mover:
                    if 'spe' in e:
                        o_act_current_speed = o_act_base_speed*boosts_dict[o_act_speed_boosts]
                        if play_para_flag:
                            p_act_current_speed = max(1,p_act_current_speed//4)
                    elif 'atk' in e:
                        o_act_current_attack = o_base_attack*boosts_dict[o_act_attack_boosts]
                        if play_brn_flag:
                            p_act_current_attack = max(1,p_act_current_attack//2)
                elif mover == active_agent.player_num and target == active_agent.opponent_num:
                    if 'spe' in e:
                        o_act_current_speed = o_act_base_speed*boosts_dict[o_act_speed_boosts]
                        if opp_para_flag:
                            o_act_current_speed = max(1,o_act_current_speed//4)
                    elif 'atk' in e:
                        o_act_current_attack = o_act_base_attack*boosts_dict[o_act_attack_boosts]
                        if opp_brn_flag:
                            o_act_current_attack = max(1,o_act_current_attack//2)
                elif mover == active_agent.player_num and target == mover:
                    if 'spe' in e:
                        p_act_current_speed = p_act_base_speed*boosts_dict[p_act_speed_boosts]
                        if opp_para_flag:
                            o_act_current_speed = max(1,o_act_current_speed//4)
                    elif 'atk' in e:
                        p_act_current_attack = p_act_base_attack*boosts_dict[o_act_attack_boosts]
                        if opp_brn_flag:
                            o_act_current_attack = max(1,o_act_current_attack//2)
        if o_act_array.size == 0:
            o_act_array = create_pkm_array(spcs=o_act_dex_num,player=1,active=1,curr_status=o_act_stat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                            toxic_counter=toxic_num,curr_health=o_act_hp_stat,max_health=o_act_max_hp,base_speed=o_act_base_speed,curr_speed=o_act_current_speed,
                            curr_special=o_act_current_special,base_attack=o_act_base_attack,curr_attack=o_act_current_attack,curr_defense=o_act_current_defense,level=100,
                            speed_boosts=o_act_speed_boosts,special_boosts=o_act_special_boosts,attack_boosts=o_act_attack_boosts,defense_boosts=o_act_defense_boosts,accuracy_boosts=o_act_accuracy_boosts,
                            reflghtscreen=0,mv1n=o_act_move_names[0],mv1pp=o_act_move_pp[0],mv2n=o_act_move_names[1],mv2pp=o_act_move_pp[1],mv3n=o_act_move_names[2],mv3pp=o_act_move_pp[2],
                            mv4n=o_act_move_names[3],mv4pp=o_act_move_pp[3],substitute_hp=0).reshape(1,-1)
        if p_act_array.size == 0:
            p_act_array = create_pkm_array(spcs=p_act_dex_num,player=1,active=1,curr_status=p_act_stat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                                        toxic_counter=toxic_num,curr_health=p_act_hp_stat,max_health=p_act_max_hp,base_speed=p_act_base_speed,curr_speed=p_act_current_speed,
                                        curr_special=p_act_current_special,base_attack=p_act_base_attack,curr_attack=p_act_current_attack,curr_defense=p_act_current_defense,level=100,
                                        speed_boosts=p_act_speed_boosts,special_boosts=p_act_special_boosts,attack_boosts=p_act_attack_boosts,defense_boosts=p_act_defense_boosts,accuracy_boosts=p_act_accuracy_boosts,
                                        reflghtscreen=0,mv1n=p_act_move_names[0],mv1pp=p_act_move_pp[0],mv2n=p_act_move_names[1],mv2pp=p_act_move_pp[1],mv3n=p_act_move_names[2],mv3pp=p_act_move_pp[2],
                                        mv4n=p_act_move_names[3],mv4pp=p_act_move_pp[3],substitute_hp=0).reshape(1,-1)
        player_array_list.insert(active_agent.player_active_index,p_act_array)

        vec_dict = {vec[0][0]: vec for vec in player_array_list}
        
        ordered_player_array_list = [vec_dict[num] for num in active_agent.original_mon_order]
        player_array = np.vstack(ordered_player_array_list)
        opp_array = np.vstack(opp_array_list)
        # player_array = np.insert(player_array,active_agent.player_active_index,p_act_array,axis=0)
        opp_array = np.insert(opp_array,active_agent.opponent_active_index,o_act_array,axis=0)

        full_array = np.concatenate((player_array,opp_array),axis=0)
        active_agent.prev_embed_battle = full_array.copy()

    def step(self, actions: Dict[str, ActionType]) -> Tuple[
        Dict[str, ObsType],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:  



        assert self.battle1 is not None

        assert self.battle2 is not None

        assert not self.battle1.finished

        assert not self.battle2.finished


        if self.agent1_to_move:
            self.agent1_to_move = False
            order1 = self.action_to_order(
                actions[self.agents[0]],
                self.battle1,
                fake=self.fake,
                strict=self.strict,
            )
            self.agent1.order_queue.put(order1)
        if self.agent2_to_move:
            self.agent2_to_move = False
            order2 = self.action_to_order(
                actions[self.agents[1]],
                self.battle2,
                fake=self.fake,
                strict=self.strict,
            )
            self.agent2.order_queue.put(order2)
        battle1 = self.agent1.battle_queue.race_get(
            self.agent1._waiting, self.agent2._trying_again
        )
        battle2 = self.agent2.battle_queue.race_get(
            self.agent2._waiting, self.agent1._trying_again
        )
        self.agent1_to_move = battle1 is not None
        self.agent2_to_move = battle2 is not None
        self.agent1._waiting.clear()
        self.agent2._waiting.clear()
        if battle1 is None:
            self.agent2._trying_again.clear()
            battle1 = self.battle1
        if battle2 is None:
            self.agent1._trying_again.clear()
            battle2 = self.battle2
        observations = {
            self.agents[0]: self.embed_battle(battle1),
            self.agents[1]: self.embed_battle(battle2),
        }
        reward = {
            self.agents[0]: self.calc_reward(battle1),
            self.agents[1]: self.calc_reward(battle2),
        }
        term1, trunc1 = self.calc_term_trunc(battle1)
        term2, trunc2 = self.calc_term_trunc(battle2)
        terminated = {self.agents[0]: term1, self.agents[1]: term2}
        truncated = {self.agents[0]: trunc1, self.agents[1]: trunc2}
        if battle1.finished:
            self.agents = []
        return observations, reward, terminated, truncated, self.get_additional_info()
    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, victory_value=1
        )


    def action_to_order(self,action: ActionType, battle: Any, fake: bool = False, strict: bool = True
    ) -> BattleOrder:
        """
        Returns the BattleOrder relative to the given action.

        :param action: The action to take.
        :type action: ActionType
        :param battle: The current battle state
        :type battle: AbstractBattle
        :param fake: If true, action-order converters will try to avoid returning a default
            output if at all possible, even if the output isn't a legal decision. Defaults
            to False.
        :type fake: bool
        :param strict: If true, action-order converters will throw an error if the move is
            illegal. Otherwise, it will return default. Defaults to True.
        :type strict: bool

        :return: The battle order for the given action in context of the current battle.
        :rtype: BattleOrder
        """

        valid_options = battle.available_switches+battle.available_moves
        name = battle.player_username

            
        if name == self.agent1.username:

            active_agent = self.agent1
        else:
            active_agent = self.agent2
        if battle.turn == 1:
            active_agent.get_full_team(battle)        
        if action == 10:
            move = Move('recharge',1)
            if move in battle.available_moves:
                return BattleOrder(move)
            else:
                return BattleOrder(random.choice(valid_options))
                # raise ValueError('Invalid Action')
        if action == 11:
            move = Move('struggle',1)
            if move in battle.available_moves:
                return BattleOrder(move)
            else:
                return BattleOrder(random.choice(valid_options))
                # raise ValueError('Invalid Action')

        if action in [0,1,2,3,4,5]:
            if action == active_agent.player_active_index:
                return BattleOrder(random.choice(valid_options))
                # raise ValueError('Illegal Action')
            switches = active_agent.full_team
            chosen_switch = switches[action]
            if chosen_switch.fainted:
                return BattleOrder(random.choice(valid_options))
                # raise ValueError('Illegal Action')
            return BattleOrder(chosen_switch)
        action -=6
        aval_moves = battle.available_moves
        
        active_mon_string = battle.active_pokemon.species

        move_list = active_agent.teaminfo[active_mon_string]
        if action > len(aval_moves)-1:
            return BattleOrder(random.choice(valid_options))
            # raise ValueError('Illegal Action')
        chosen_move = aval_moves[action]
        selected_move = move_list[action]

        if chosen_move == selected_move:
            return BattleOrder(chosen_move)
        else:
            return BattleOrder(random.choice(valid_options))
            # raise ValueError('Invalid Action')       
            
        
    def render_mode(self, mode = "human"):
        return super().render(mode)


    def order_to_action(
        order: BattleOrder, battle: Any, fake: bool = False, strict: bool = True
    ) -> ActionType:
        """
        Returns the action relative to the given BattleOrder.

        :param order: The order to take.
        :type order: BattleOrder
        :param battle: The current battle state
        :type battle: AbstractBattle
        :param fake: If true, action-order converters will try to avoid returning a default
            output if at all possible, even if the output isn't a legal decision. Defaults
            to False.
        :type fake: bool
        :param strict: If true, action-order converters will throw an error if the move is
            illegal. Otherwise, it will return default. Defaults to True.
        :type strict: bool

        :return: The action for the given battle order in context of the current battle.
        :rtype: ActionType
        """
        pass
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, ObsType], Dict[str, Dict[str, Any]]]:
        self.agents = [self.agent1.username, self.agent2.username]

        
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        if options:
            if 'agent1' in options:

                self.agent1.update_team(options['agent1'])
            else:

                newteam1 = team_generator_alt(seed_num=seed)
                self.agent1.update_team(newteam1)
            if 'agent2' in options:

                self.agent2.update_team(options['agent2'])
            else:

                newteam2 = team_generator_alt(seed_num=seed)
                self.agent2.update_team(newteam2)
        elif options is None:
            newteam1 = team_generator_alt(seed_num=seed)

            newteam2 = team_generator_alt(seed_num=seed)

            self.agent1.update_team(newteam1)

            self.agent2.update_team(newteam2)   

        if not self.agent1.battle or not self.agent2.battle:
            count = self._INIT_RETRIES
            while not self.agent1.battle or not self.agent2.battle:
                if count == 0:
                    raise RuntimeError("Agent is not challenging")
                count -= 1
                time.sleep(self._TIME_BETWEEN_RETRIES)
        if self.battle1 and not self.battle1.finished:
            assert self.battle2 is not None
            if self.battle1 == self.agent1.battle:
                if self.agent1_to_move:
                    self.agent1_to_move = False
                    self.agent1.order_queue.put(ForfeitBattleOrder())
                    if self.agent2_to_move:
                        self.agent2_to_move = False
                        self.agent2.order_queue.put(DefaultBattleOrder())
                else:
                    assert self.agent2_to_move
                    self.agent2_to_move = False
                    self.agent2.order_queue.put(ForfeitBattleOrder())
                self.agent1.battle_queue.get()
                self.agent2.battle_queue.get()
            else:
                raise RuntimeError(
                    "Environment and agent aren't synchronized. Try to restart"
                )
        self.battle1 = self.agent1.battle_queue.get()
        self.battle2 = self.agent2.battle_queue.get()
        self.agent1_to_move = True
        self.agent2_to_move = True
        observations = {
            self.agents[0]: self.embed_battle(self.battle1),
            self.agents[1]: self.embed_battle(self.battle2),
        }
        return observations, self.get_additional_info()

    

env = Pokebot_Gen1_Environment(
    battle_format="gen1ou", 
    team1= team_1,
    team2= team_2,
    start_challenging=True 
)
# obs, info = env.reset()
# print("Initial observations:", obs)
# print("Info:", info)
# from pettingzoo.test import parallel_api_test
# parallel_api_test(env,num_cycles=100)