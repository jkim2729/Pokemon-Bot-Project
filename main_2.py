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


def create_pkm_array(spcs = 0, player = 1, active = 0, curr_status = 0, add_status = 0, partial_trap = 0, toxic_counter = 1, curr_health = 100, max_health = 100, 
                base_speed = 0,curr_speed = 0, curr_special = 0, base_attack = 0, curr_attack = 0, curr_defense = 0, level = 0, 
                speed_boosts = 7, special_boosts = 7, attack_boosts = 7, defense_boosts = 7, accuracy_boosts = 7,
                reflghtscreen = 0, mv1n = 1, mv1pp = 100, mv2n = 1, mv2pp = 100, mv3n = 1, mv3pp = 100, mv4n = 1,mv4pp = 100,substitute_hp = 0):
            return np.array([spcs,player,active,curr_status,add_status,partial_trap,toxic_counter,curr_health,max_health,base_speed,curr_speed,curr_special,base_attack,curr_attack,
                        curr_defense,level,speed_boosts,special_boosts,attack_boosts,defense_boosts,accuracy_boosts,reflghtscreen,
                        mv1n,mv1pp,mv2n,mv2pp,mv3n,mv3pp,mv4n,mv4pp,substitute_hp],dtype= np.uint16)

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
        player_array = np.array([])
        opp_array = np.array([])
        if battle.turn == 1:
            player_team = battle.team
            opponent_team = battle.opponent_team
            partial_trap = 0
            toxic_counter = 0
            alt_status = 0
            sub_hp = 0
            index = 0
            zipped = zip_longest(player_team,opponent_team,fillvalue=UnknownMon())
            for pmon,omon in zipped:
                if isinstance(omon,UnknownMon):
                    omon_array = omon.info
                else:
                    ospecies = omon.species
                    odex_num = dex_nums[ospecies]
                    o_act_base_speed = stats[ospecies]['Speed_Total']
                    o_act_current_speed = o_act_base_speed
                    o_act_current_special = stats[ospecies]['Special_Total']
                    o_act_base_attack = stats[ospecies]['Attack_Total']
                    o_act_current_attack = o_act_base_attack
                    o_act_current_defense = stats[ospecies]['Defense_Total']
                    omon_array = create_pkm_array(spcs = odex_num, player=1,active=1,curr_status=0,add_status=0,partial_trap=0,
                                                  toxic_counter=1,curr_health=omon.current_hp,max_health=omon.max_hp,base_speed=o_act_base_speed,curr_speed=o_act_current_speed,
                                                  curr_special=o_act_current_special,base_attack=o_act_base_attack,curr_attack=o_act_current_attack,curr_defense=o_act_current_defense,
                                                  level=100,speed_boosts = 7, special_boosts = 7, attack_boosts = 7, defense_boosts = 7, accuracy_boosts = 7,
                                                reflghtscreen = 0, mv1n = 1, mv1pp = 100, mv2n = 1, mv2pp = 100, mv3n = 1, mv3pp = 100, mv4n = 1,mv4pp = 100,substitute_hp = 0)
                opp_array = np.vstack((opp_array, omon_array)) if opp_array.size > 0 else omon_array
                p_stats = pmon.stats

                if pmon.active:
                    pactive_num = 1
                else:
                    pactive_num = 0
                pspecies = pmon.species
                pdex_num = dex_nums[pspecies]
                pmoves = pmon.moves
                move_names = []
                move_pp = []
                for k,v in pmoves.items():
                    move_names.append(rev_move_dict[k])
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
                                              mv4n=move_names[3],mv4pp=move_pp[3],substitute_hp=0
                                              )
                player_array = np.vstack((player_array, pmon_array)) if player_array.size > 0 else pmon_array
            full_array = np.vstack((player_array,opp_array))
            return full_array
        player_team = battle.team
        
        opponent_team = battle.opponent_team
        partial_trap_num = 0
        toxic_num = 1
        alt_status = 0
        sub_hp = 0 
        player_mon = battle.active_pokemon
        opponent_mon = battle.opponent_active_pokemon
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
        player_status = player_mon.status.value
        if player_status == 1:
            play_brn_flag = True
        elif player_status == 4:
            play_para_flag = True
        
        player_array = np.zeros(shape=(12,26),dtype=np.uint16)
        opp_array = np.zeros(shape=(12,26),dtype=np.uint16)
        observations = battle.observations
        
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
        zipped = zip_longest(player_team,opponent_team,fillvalue=0)
        for pmon,omon in zipped:
            pspecies = pmon.species
            ospecies = omon.species
            pdex_num = dex_nums[pspecies]
            odex_num = dex_nums[ospecies]
            pteam_num = 0
            oteam_num = 1
            p_stats = pmon.stats
            
            mover = None
            if pmon.active:
                pactive_num = 1
                p_act_dex_num = pdex_num
                p_act_base_speed = p_stats['spe']
                p_act_base_attack = p_stats['atk']
                p_act_current_special = p_stats['spd']
                p_act_current_defense = p_stats['def']
                p_boost_dict = pmon.boosts
                p_act_accuracy_boosts = 7 + p_boost_dict['acc']
                p_act_speed_boosts = 7 + p_boost_dict['spe']
                p_act_attack_boosts = 7 + p_boost_dict['atk']
                p_act_special_boosts = 7 + p_boost_dict['spd']
                p_act_defense_boosts = 7 + p_boost_dict['def']
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
                    p_act_move_names.append(rev_move_dict[k])
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
            if omon.active:
                oactive_num = 1
                o_act_dex_num = odex_num
                o_boost_dict = omon.boosts
                o_act_accuracy_boosts = 7 + o_boost_dict['acc']
                o_act_speed_boosts = 7 + o_boost_dict['spe']
                o_act_attack_boosts = 7 + o_boost_dict['atk']
                o_act_special_boosts = 7 + o_boost_dict['spd']
                o_act_defense_boosts = 7 + o_boost_dict['def']
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
                    o_act_move_names.append(rev_move_dict[k])
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
                    o_act_base_speed = stats[ospecies]['Speed_Total']
                    o_act_current_speed = o_act_base_speed*boosts_dict[o_act_speed_boosts]
                    if opp_para_flag:
                        o_act_current_speed = max(1,o_act_current_speed//4)
                        
                else:
                    o_act_base_speed = self.prev_embed_battle[index+6][9]
                    o_act_current_speed = self.prev_embed_battle[index+6][10]
                if self.prev_embed_battle[index+6][11] == 0:
                    o_act_current_special = stats[ospecies]['Special_Total']
                else:
                    o_act_current_special = self.prev_embed_battle[index+6][11]
                if self.prev_embed_battle[index+6][12] == 0:
                    o_act_base_attack = stats[ospecies]['Attack_Total']
                    o_act_current_attack = o_act_base_attack
                if self.prev_embed_battle[index+6][14] == 0:
                    o_act_current_defense = stats[ospecies]['Defense_Total']




            else:
                oactive_num = 0
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
                    move_names.append(rev_move_dict[k])
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
                                              mv4n=move_names[3],mv4pp=move_pp[3],substitute_hp=0
                                               )
                player_array = np.vstack((player_array, p_mon_array)) if player_array.size > 0 else p_mon_array
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
                    move_names.append(rev_move_dict[k])
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
                                              mv4n=move_names[3],mv4pp=move_pp[3],substitute_hp=0
                                               )
                opp_array = np.vstack((opp_array, o_mon_array)) if opp_array.size > 0 else o_mon_array        
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
                                              mv4n=o_act_move_names[3],mv4pp=o_act_move_pp[3],substitute_hp=0)
                else:
                    p_act_array = create_pkm_array(spcs=p_act_dex_num,player=1,active=1,curr_status=p_act_stat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                                               toxic_counter=toxic_num,curr_health=p_act_hp_stat,max_health=p_act_max_hp,base_speed=p_act_base_speed,curr_speed=p_act_current_speed,
                                               curr_special=p_act_current_special,base_attack=p_act_base_attack,curr_attack=p_act_current_attack,curr_defense=p_act_current_defense,level=100,
                                               speed_boosts=p_act_speed_boosts,special_boosts=p_act_special_boosts,attack_boosts=p_act_attack_boosts,defense_boosts=p_act_defense_boosts,accuracy_boosts=p_act_accuracy_boosts,
                                               reflghtscreen=0,mv1n=p_act_move_names[0],mv1pp=p_act_move_pp[0],mv2n=p_act_move_names[1],mv2pp=p_act_move_pp[1],mv3n=p_act_move_names[2],mv3pp=p_act_move_pp[2],
                                              mv4n=p_act_move_names[3],mv4pp=p_act_move_pp[3],substitute_hp=0)
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
                        p_act_current_speed = p_act_base_speed*boosts_dict[p_act_speed_boosts]
                        if play_para_flag:
                            p_act_current_speed = p_act_current_speed//4
                    elif 'atk' in e:
                        p_act_current_attack = p_act_base_attack*boosts_dict[p_act_attack_boosts]
                        if play_brn_flag:
                            p_act_current_attack = p_act_current_attack//2
                elif mover == self.opponent_num and target == mover:
                    if 'spe' in e:
                        o_act_current_speed = o_act_base_speed*boosts_dict[o_act_speed_boosts]
                        if play_para_flag:
                            p_act_current_speed = p_act_current_speed//4
                    elif 'atk' in e:
                        o_act_current_attack = o_base_attack*boosts_dict[o_act_attack_boosts]
                        if play_brn_flag:
                            p_act_current_attack = p_act_current_attack//2
                elif mover == self.player_num and target == self.opponent_num:
                    if 'spe' in e:
                        o_act_current_speed = o_act_base_speed*boosts_dict[o_act_speed_boosts]
                        if opp_para_flag:
                            o_act_current_speed = o_act_current_speed//4
                    elif 'atk' in e:
                        o_act_current_attack = o_act_base_attack*boosts_dict[o_act_attack_boosts]
                        if opp_brn_flag:
                            o_act_current_attack = o_act_current_attack//2
                elif mover == self.player_num and target == mover:
                    if 'spe' in e:
                        p_act_current_speed = p_act_base_speed*boosts_dict[p_act_speed_boosts]
                        if opp_para_flag:
                            o_act_current_speed = o_act_current_speed//4
                    elif 'atk' in e:
                        p_act_current_attack = p_act_base_attack*boosts_dict[o_act_attack_boosts]
                        if opp_brn_flag:
                            o_act_current_attack = o_act_current_attack//2
        if o_act_array.size == 0:
            o_act_array = create_pkm_array(spcs=o_act_dex_num,player=1,active=1,curr_status=o_act_stat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                            toxic_counter=toxic_num,curr_health=o_act_hp_stat,max_health=o_act_max_hp,base_speed=o_act_base_speed,curr_speed=o_act_current_speed,
                            curr_special=o_act_current_special,base_attack=o_act_base_attack,curr_attack=o_act_current_attack,curr_defense=o_act_current_defense,level=100,
                            speed_boosts=o_act_speed_boosts,special_boosts=o_act_special_boosts,attack_boosts=o_act_attack_boosts,defense_boosts=o_act_defense_boosts,accuracy_boosts=o_act_accuracy_boosts,
                            reflghtscreen=0,mv1n=o_act_move_names[0],mv1pp=o_act_move_pp[0],mv2n=o_act_move_names[1],mv2pp=o_act_move_pp[1],mv3n=o_act_move_names[2],mv3pp=o_act_move_pp[2],
                            mv4n=o_act_move_names[3],mv4pp=o_act_move_pp[3],substitute_hp=0)
        if p_act_array.size == 0:
            p_act_array = create_pkm_array(spcs=p_act_dex_num,player=1,active=1,curr_status=p_act_stat_condition,add_status=alt_status,partial_trap=partial_trap_num,
                                        toxic_counter=toxic_num,curr_health=p_act_hp_stat,max_health=p_act_max_hp,base_speed=p_act_base_speed,curr_speed=p_act_current_speed,
                                        curr_special=p_act_current_special,base_attack=p_act_base_attack,curr_attack=p_act_current_attack,curr_defense=p_act_current_defense,level=100,
                                        speed_boosts=p_act_speed_boosts,special_boosts=p_act_special_boosts,attack_boosts=p_act_attack_boosts,defense_boosts=p_act_defense_boosts,accuracy_boosts=p_act_accuracy_boosts,
                                        reflghtscreen=0,mv1n=p_act_move_names[0],mv1pp=p_act_move_pp[0],mv2n=p_act_move_names[1],mv2pp=p_act_move_pp[1],mv3n=p_act_move_names[2],mv3pp=p_act_move_pp[2],
                                        mv4n=p_act_move_names[3],mv4pp=p_act_move_pp[3],substitute_hp=0)

        player_array = np.insert(player_array,self.player_active_index,p_act_array)
        opp_array = np.insert(opp_array,self.opponent_active_index,o_act_array)
        full_array = np.vstack((player_array,opp_array))
        return full_array                                                     
                                





            

            


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
        
