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
          Attack boosts, Defense boosts, Accuracy boosts, Reflect-LightScreen, Move 1 Name, Move 1 PP, Move 2 Name, Move 2 PP, 
          Move 3 Name, Move 3 PP, Move 4 Name, Move 4 PP,prev_dmg (used for counter),substitute_hp (0 if no substitute is in place)]
          
        '''
        
        self._p1_data = np.zeros(shape=(12,32),dtype=np.uint16)
        self._p2_data = np.zeros(shape=(12,32),dtype=np.uint16)
        self._finished = None
        self._battle_array = np.zeros(shape=(12,32),dtype=np.uint16)
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

    def create_pkm_array(self,spcs = 0, player = 0, active = 0, curr_status = 0, add_status = 0, partial_trap = 0, toxic_counter = 1, curr_health = 1000, max_health = 1000, 
                         base_speed = 0,curr_speed = 0, curr_special = 0, base_attack = 0, curr_attack = 0, curr_defense = 0, level = 0, 
                         speed_boosts = 0, special_boosts = 0, attack_boosts = 0, defense_boosts = 0, accuracy_boosts = 0,
                         reflghtscreen = 0, mv1n = 1, mv1pp = 100, mv2n = 1, mv2pp = 100, mv3n = 1, mv3pp = 100, mv4n = 1,mv4pp = 100,substitute_hp = 0):
        
        return np.array([spcs,player,active,curr_status,add_status,partial_trap,toxic_counter,curr_health,max_health,base_speed,curr_speed,curr_special,base_attack,curr_attack,
                         curr_defense,level,speed_boosts,special_boosts,attack_boosts,defense_boosts,accuracy_boosts,reflghtscreen,
                         mv1n,mv1pp,mv2n,mv2pp,mv3n,mv3pp,mv4n,mv4pp,substitute_hp],dtype= np.uint16)

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



    def reg_move(self,player,rows,move,clamp_flag): #partial trapping moves need to be treated differently so we have clamp flag
        move_data = Move(move,1) #use dict before function to convert num to string
        move_index = rev_move_dict[move_data]
        flinch_status = False
        clamp_status = False
        player_fnt = False
        opp_fnt = False
        if player == 0:
            pkm_array = self.p1_data[rows[0]]
            play_view_opp_array = self.p1_data[6+rows[1]]
            opp_view_pkm_array = self.p2_data[6+rows[0]]
            opp_pkm_array = self.p2_data[rows[1]]
        elif player == 1:
            pkm_array = self.p2_data[rows[0]]
            opp_pkm_array = self.p1_data[rows[1]]

        else:
            raise ValueError('Unknown Value for player')

        if move_index == pkm_array[22]:
            array_location = pkm_array[22]
        elif move_index == pkm_array[24]:
            array_location = pkm_array[24]
        elif move_index == pkm_array[26]:
            array_location = pkm_array[26]
        elif move_index == pkm_array[28]:
            array_location = pkm_array[28]
        else:
            raise ValueError('Invalid Move')



        ### check to see if we actually get to use a move
        pkm_status = pkm_array[3]
        if pkm_status in set([6,7,8,9,10,11]): #sleep

            wakeup_chance = (1)/(13-pkm_status)
            sleep_num = random.random()
            if wakeup_chance>sleep_num:
                pkm_array[3] = 2
                opp_view_pkm_array = 2
            else:
                opp_view_pkm_array+=1
                pkm_array[3]+=1
            return (player_fnt,opp_fnt,flinch_status,clamp_status)
        elif pkm_status == 12: #last turn of sleep
            pkm_array[3] = 2
            opp_view_pkm_array = 2
            return (player_fnt,opp_fnt,flinch_status,clamp_status)

        elif pkm_status == 4: #paralysis
            para_num = random.random()
            if para_num <0.25:
                
                return (player_fnt,opp_fnt,flinch_status,clamp_status)
        elif pkm_status == 3: #freeze

            return (player_fnt,opp_fnt,flinch_status,clamp_status)
        alt_status_num =opp_pkm_array[4]
        if 'confusion' in alt_status_dict[alt_status_num]:
            confusion_num = random.random()
            if confusion_num<0.5:
                if pkm_array[21] == 1 or pkm_array[21] == 3:
                        reflect_mod = 2
                else:
                        reflect_mod = 1
                confusion_dmg = confusion_dmg(level = pkm_array[15],atk=pkm_array[13],dfs=opp_pkm_array[14],
                                                  atk_mod=boosts_dict[pkm_array[18]],dfs_mod= boosts_dict[pkm_array[19]])
                if opp_pkm_array[-1]> 0 and pkm_array[-1]>0:
                    opp_sub_hp = opp_pkm_array[-1]
                    opp_sub_hp = max(0,opp_sub_hp-confusion_dmg)
                    play_view_opp_array[-1] = opp_sub_hp
                    opp_pkm_array[-1] = opp_sub_hp
                    return (player_fnt,opp_fnt,flinch_status,clamp_status)
                elif pkm_array[-1]>0:
                    confusion_dmg = 0
                curr_health = pkm_array[7]
                curr_health = max(0,curr_health-confusion_dmg)
                pkm_array[7] = curr_health
                opp_view_pkm_array[7] = curr_health
                if curr_health>0:
                    return (player_fnt,opp_fnt,flinch_status,clamp_status)
                else:
                    opp_view_pkm_array[3] = 2
                    pkm_array[3] = 2
                    opp_fnt = True
                    return (player_fnt,opp_fnt,flinch_status,clamp_status)


        move_type = move.type.name
        move_power = move_data.base_power
        if move_type!= 'NORMAL' or move_type!= 'FIGHTING' or move_power == 0:
            opp_pkm_array[-2] = 0
            play_view_opp_array[-2] = 0

        if move not in ['clamp','wrap','bind','firespin']:
            opp_pkm_array[5] = 0
            
            opp_view_pkm_array[array_location+1] -=1
            pkm_array[array_location+1] -=1
        if clamp_flag:
            opp_view_pkm_array[array_location+1] -=1
            pkm_array[array_location+1] -=1
        if move in loaded_data['full_acc_moves']:
            accuracy = 1
        else:
            accuracy = move_data.accuracy
            accuracy*=(255/256)
            accuracy*=boosts_dict[pkm_array[20]]
        acc_num = random.random()
        
        if acc_num> accuracy: #move missed
            return (player_fnt,opp_fnt,flinch_status,clamp_status)
        move_category = move_data.category.value
        if move_category in [1,2]:

            type_modifier = 1
            opp_pokemon = rev_dex_nums[opp_pkm_array[0]]
            player_pkm = rev_dex_nums[pkm_array[0]]
            for type in pokemon_types[opp_pokemon]:
                t = TypeChart[move_type]
                type_modifier*= t[type]
            stab_mod = 1
            if move_type in pokemon_types[player_pkm]:
                stab_mod = 1.5


            
            if move == 'counter':

                dmg_amount = 2*pkm_array[-2]
                pkm_array[-2] = 0
                opp_view_pkm_array[-2] = 0
            elif move == 'superfang':
                dmg_amount = opp_pkm_array[7]//2
            elif move_data.damage>0:
                if move_data.damage == 'level':
                    dmg_amount = opp_pkm_array[15]
                else:
                    dmg_amount = move_data.damage
            
            
            elif move in ['clamp','wrap','bind','firespin']:
                clamp_status = True
                partial_trap_str = partial_trapped_dict[opp_pkm_array[5]]
                if partial_trap_str == 'healthy':
                    if move_category == 1:
                        if pkm_array[21] == 1 or pkm_array[21] == 3:
                            reflect_mod = 2
                        else:
                            reflect_mod = 1
                        dmg_amount = select_dmg_value(level = pkm_array[15],speed=pkm_array[9],power = move_power,atk=pkm_array[13],dfs=opp_pkm_array[14],
                                                        atk_mod=boosts_dict[pkm_array[18]],dfs_mod= reflect_mod*boosts_dict[opp_pkm_array[19]],stab= stab_mod, type_mult= type_modifier)
                    elif move_category == 2:
                        if pkm_array[21] == 2 or pkm_array[21] == 3:
                            reflect_mod = 2
                        else:
                            reflect_mod = 1
                        dmg_amount = select_dmg_value(level = pkm_array[15],speed=pkm_array[9],power = move_power,atk=pkm_array[11],dfs=opp_pkm_array[11],
                                                    atk_mod=boosts_dict[pkm_array[17]],dfs_mod= reflect_mod*boosts_dict[opp_pkm_array[17]],stab= stab_mod, type_mult= type_modifier)
                else:
                    clamp_crit = partial_trap_str[0]
                    if move_category == 1:
                        if pkm_array[21] == 1 or pkm_array[21] == 3:
                            reflect_mod = 2
                        else:
                            reflect_mod = 1
                        dmg_amount = select_dmg_value(level = pkm_array[15],speed=pkm_array[9],power = move_power,atk=pkm_array[13],dfs=opp_pkm_array[14],
                                                        atk_mod=boosts_dict[pkm_array[18]],dfs_mod= reflect_mod*boosts_dict[opp_pkm_array[19]],stab= stab_mod, type_mult= type_modifier,crit_status=clamp_crit)
                    elif move_category == 2:
                        if pkm_array[21] == 2 or pkm_array[21] == 3:
                            reflect_mod = 2
                        else:
                            reflect_mod = 1
                        dmg_amount = select_dmg_value(level = pkm_array[15],speed=pkm_array[9],power = move_power,atk=pkm_array[11],dfs=opp_pkm_array[11],
                                                    atk_mod=boosts_dict[pkm_array[17]],dfs_mod= reflect_mod*boosts_dict[opp_pkm_array[17]],stab= stab_mod, type_mult= type_modifier,crit_status=clamp_crit)

            elif move == 'explosion' or 'selfdestruct':
                if pkm_array[21] == 1 or pkm_array[21] == 3:
                    reflect_mod = 2
                else:
                    reflect_mod = 1
                dmg_amount = select_dmg_value(level = pkm_array[15],speed=pkm_array[9],power = move_power,atk=pkm_array[13],dfs=opp_pkm_array[14]//2,
                                                atk_mod=boosts_dict[pkm_array[18]],dfs_mod= reflect_mod*boosts_dict[opp_pkm_array[19]],stab= stab_mod, type_mult= type_modifier)
                pkm_array[7] = 0 
                opp_view_pkm_array[7] = 0                    

            elif move_category == 1:
                if pkm_array[21] == 1 or pkm_array[21] == 3:
                    reflect_mod = 2
                else:
                    reflect_mod = 1
                dmg_amount = select_dmg_value(level = pkm_array[15],speed=pkm_array[9],power = move_power,atk=pkm_array[13],dfs=opp_pkm_array[14],
                                                atk_mod=boosts_dict[pkm_array[18]],dfs_mod= reflect_mod*boosts_dict[opp_pkm_array[19]],stab= stab_mod, type_mult= type_modifier)
            elif move_category == 2:
                if pkm_array[21] == 2 or pkm_array[21] == 3:
                    reflect_mod = 2
                else:
                    reflect_mod = 1
                dmg_amount = select_dmg_value(level = pkm_array[15],speed=pkm_array[9],power = move_power,atk=pkm_array[11],dfs=opp_pkm_array[11],
                                                atk_mod=boosts_dict[pkm_array[17]],dfs_mod= reflect_mod*boosts_dict[opp_pkm_array[17]],stab= stab_mod, type_mult= type_modifier)
                if move_type == 'FIRE' and dmg_amount>0 and opp_pkm_array[3] == 3:
                    opp_pkm_array[3] = 0

            if move_type == 'NORMAL' or move_type == 'FIGHTING':
                if dmg_amount>0:
                    opp_pkm_array[-2] = dmg_amount
                    play_view_opp_array[-2] = dmg_amount
            if dmg_amount == 0:
                pkm_array[-2] = 0
                opp_view_pkm_array[-2] = 0

            
            if move_data.secondary:
                sec_dict = move_data.secondary
                chance = sec_dict['chance']
                sec_num = random.random()
                if sec_num<chance:
                    if 'boosts' in sec_dict:
                        move_boosts = sec_dict['boosts']
                
                        if 'spe' in move_boosts:
                            
                            boost_num = move_boosts['spe']
                            if boost_num>0 and opp_pkm_array == 13:
                                return (player_fnt,opp_fnt,flinch_status,clamp_status)
                            elif boost_num< 0 and opp_pkm_array == 1:
                                return (player_fnt,opp_fnt,flinch_status,clamp_status)
                            
                            opp_pkm_array[16] = boost_num+opp_pkm_array[16]
                            opp_pkm_array[10] = min(999,max(1,math.floor(boosts_dict[opp_pkm_array[16]]*opp_pkm_array[9])))
                            play_view_opp_array[16] = opp_pkm_array[16]
                            play_view_opp_array[10] = min(999,max(1,math.floor(boosts_dict[play_view_opp_array[16]]*play_view_opp_array[9])))
                            if opp_pkm_array[3] == 1:
                                opp_pkm_array[13]//=2
                                play_view_opp_array[13]//=2
                            elif opp_pkm_array[3] == 4:
                                opp_pkm_array[10]//=4
                                play_view_opp_array[10]//=4
                        elif 'spd' in move_boosts:
                            boost_num = move_boosts['spd']
                            if boost_num>0 and opp_pkm_array == 13:
                                return (player_fnt,opp_fnt,flinch_status,clamp_status)
                            elif boost_num< 0 and opp_pkm_array == 1:
                                return (player_fnt,opp_fnt,flinch_status,clamp_status)
                            
                            opp_pkm_array[17] = boost_num+opp_pkm_array[17]
                            play_view_opp_array[17] = opp_pkm_array[17]
                            if opp_pkm_array[3] == 1:
                                opp_pkm_array[13]//=2
                                play_view_opp_array[13]//=2
                            elif opp_pkm_array[3] == 4:
                                opp_pkm_array[10]//=4
                                play_view_opp_array[10]//=4

                        elif 'atk' in move_boosts:
                            
                            boost_num = move_boosts['atk']
                            if boost_num>0 and opp_pkm_array == 13:
                                return (player_fnt,opp_fnt,flinch_status,clamp_status)
                            elif boost_num< 0 and opp_pkm_array == 1:
                                return (player_fnt,opp_fnt,flinch_status,clamp_status)
                            
                            opp_pkm_array[18] = boost_num+opp_pkm_array[18]
                            opp_pkm_array[13] = min(999,max(1,math.floor(boosts_dict[opp_pkm_array[18]]*opp_pkm_array[12])))
                            play_view_opp_array[18] = opp_pkm_array[18]
                            play_view_opp_array[13] = min(999,max(1,math.floor(boosts_dict[play_view_opp_array[18]]*play_view_opp_array[12])))
                            if opp_pkm_array[3] == 1:
                                opp_pkm_array[13]//=2
                                play_view_opp_array[13]//=2
                            elif opp_pkm_array[3] == 4:
                                opp_pkm_array[10]//=4
                                play_view_opp_array[10]//=4
                        elif 'def' in move_boosts:
                            
                            boost_num = move_boosts['def']
                            if boost_num>0 and opp_pkm_array == 13:
                                return (player_fnt,opp_fnt,flinch_status,clamp_status)
                            elif boost_num< 0 and opp_pkm_array == 1:
                                return (player_fnt,opp_fnt,flinch_status,clamp_status)
                            
                            opp_pkm_array[19] = boost_num+opp_pkm_array[19]
                            play_view_opp_array[19] = opp_pkm_array[19]
                            if opp_pkm_array[3] == 1:
                                opp_pkm_array[13]//=2
                                play_view_opp_array[13]//=2
                            elif opp_pkm_array[3] == 4:
                                opp_pkm_array[10]//=4
                                play_view_opp_array[10]//=4
                    
                    if 'status' in sec_dict:
                        status_num = move_data.status.value
                        if opp_pkm_status == 0:
                            opp_pkm_array[3] == status_num
                    
                    if 'volatileStatus' in sec_dict:
                        if sec_dict['volatileStatus'] == 'confusion':
                            if alt_status_num == 0:
                                alt_status_num =2
                            elif alt_status_num == 1:
                                alt_status_num = 7
                            elif alt_status_num == 6:
                                alt_status_num = 12
                            elif alt_status_num == 11:
                                alt_status_num = 16
                            opp_pkm_array[4] = alt_status_num

                        elif sec_dict['volatilteStatus'] == 'flinch':
                            flinch_status = True
            if move_data.n_hit == (2,2):
                dmg_amount = dmg_amount*2
            elif move_data.n_hit == (2,5):
                num_hit_chance = random.random()
                if num_hit_chance < 0.375:
                    dmg_amount*=2
                elif num_hit_chance < 0.75:
                    dmg_amount*=3
                elif num_hit_chance < 0.875:
                    dmg_amount*=4
                elif num_hit_chance < 1:
                    dmg_amount*=5
            if play_view_opp_array[-1]>0:
                opp_pkm_array[-1] = max(0,opp_pkm_array[-1]-dmg_amount)
                play_view_opp_array[-1] = max(0,play_view_opp_array[-1]-dmg_amount)
            else:
                opp_pkm_array[7] = max(0,opp_pkm_array[7]-dmg_amount)
                play_view_opp_array[7] = max(0,play_view_opp_array[7]-dmg_amount)
            if move_data.recoil:
                recoil_dmg = move_data.recoil * dmg_amount
                pkm_array[7] = max(0,pkm_array[7]-recoil_dmg)
                opp_view_pkm_array[7] = max(0,play_view_opp_array[7]-recoil_dmg)    
            if opp_pkm_array[7] == 0:
                opp_pkm_array[3] = 2
                play_view_opp_array[3] = 2
                opp_fnt = True
            if pkm_array[7] == 0:
                pkm_array[3] = 2
                opp_view_pkm_array[3] = 2
                player_fnt = True
            
            return(player_fnt,opp_fnt,flinch_status,clamp_status)


        elif move_category == 3:
            pkm_array[-2] = 0
            opp_view_pkm_array[-2] = 0
            if move == 'confuseray' or move == 'supersonic':
                
                if alt_status_num == 0:
                    alt_status_num =2
                elif alt_status_num == 1:
                    alt_status_num = 7
                elif alt_status_num == 6:
                    alt_status_num = 12
                elif alt_status_num == 11:
                    alt_status_num = 16
                opp_pkm_array[4] = alt_status_num
                return (player_fnt,opp_fnt,flinch_status,clamp_status)
            elif move == 'leechseed':

                if alt_status_num == 0:
                    alt_status_num =1
                elif alt_status_num in [2,3,4,5,6]:
                    alt_status_num+=5
                elif alt_status_num in [12,13,14,15]:
                    alt_status_num+=4
                opp_pkm_array[4] = alt_status_num
                return (player_fnt,opp_fnt,flinch_status,clamp_status)

            opp_pkm_status = opp_pkm_array[3]
            if move == 'toxic':
                
                if opp_pkm_status == 0:
                    opp_pkm_array[3] == 13
                return (player_fnt,opp_fnt,flinch_status,clamp_status)
            
            status_num = move_data.status.value
            move_target = move_data.target.name
            if status_num:
                if opp_pkm_status == 0:
                    opp_pkm_array[3] == status_num
                return (player_fnt,opp_fnt,flinch_status,clamp_status)
            if move_target == 'NORMAL':
                move_boosts = move_data.boosts


                
                if 'spe' in move_boosts:
                    
                    boost_num = move_boosts['spe']
                    if boost_num>0 and opp_pkm_array == 13:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    elif boost_num< 0 and opp_pkm_array == 1:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    
                    opp_pkm_array[16] = boost_num+opp_pkm_array[16]
                    opp_pkm_array[10] = min(999,max(1,math.floor(boosts_dict[opp_pkm_array[16]]*opp_pkm_array[9])))
                    play_view_opp_array[16] = opp_pkm_array[16]
                    play_view_opp_array[10] = min(999,max(1,math.floor(boosts_dict[play_view_opp_array[16]]*play_view_opp_array[9])))
                    if opp_pkm_array[3] == 1:
                        opp_pkm_array[13]//=2
                        play_view_opp_array[13]//=2
                    elif opp_pkm_array[3] == 4:
                        opp_pkm_array[10]//=4
                        play_view_opp_array[10]//=4
                elif 'spd' in move_boosts:
                    boost_num = move_boosts['spd']
                    if boost_num>0 and opp_pkm_array == 13:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    elif boost_num< 0 and opp_pkm_array == 1:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    
                    opp_pkm_array[17] = boost_num+opp_pkm_array[17]
                    play_view_opp_array[17] = opp_pkm_array[17]
                    if opp_pkm_array[3] == 1:
                        opp_pkm_array[13]//=2
                        play_view_opp_array[13]//=2
                    elif opp_pkm_array[3] == 4:
                        opp_pkm_array[10]//=4
                        play_view_opp_array[10]//=4

                elif 'atk' in move_boosts:
                    
                    boost_num = move_boosts['atk']
                    if boost_num>0 and opp_pkm_array == 13:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    elif boost_num< 0 and opp_pkm_array == 1:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    
                    opp_pkm_array[18] = boost_num+opp_pkm_array[18]
                    opp_pkm_array[13] = min(999,max(1,math.floor(boosts_dict[opp_pkm_array[18]]*opp_pkm_array[12])))
                    play_view_opp_array[18] = opp_pkm_array[18]
                    play_view_opp_array[13] = min(999,max(1,math.floor(boosts_dict[play_view_opp_array[18]]*play_view_opp_array[12])))
                    if opp_pkm_array[3] == 1:
                        opp_pkm_array[13]//=2
                        play_view_opp_array[13]//=2
                    elif opp_pkm_array[3] == 4:
                        opp_pkm_array[10]//=4
                        play_view_opp_array[10]//=4
                elif 'def' in move_boosts:
                    
                    boost_num = move_boosts['def']
                    if boost_num>0 and opp_pkm_array == 13:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    elif boost_num< 0 and opp_pkm_array == 1:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    
                    opp_pkm_array[19] = boost_num+opp_pkm_array[19]
                    play_view_opp_array[19] = opp_pkm_array[19]
                    if opp_pkm_array[3] == 1:
                        opp_pkm_array[13]//=2
                        play_view_opp_array[13]//=2
                    elif opp_pkm_array[3] == 4:
                        opp_pkm_array[10]//=4
                        play_view_opp_array[10]//=4
            elif move_target == 'SELF':
                move_boosts = move_data.boosts


                
                if 'spe' in move_boosts:
                    
                    boost_num = move_boosts['spe']
                    if boost_num>0 and pkm_array == 13:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    elif boost_num< 0 and pkm_array == 1:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    
                    pkm_array[16] = boost_num+pkm_array[16]
                    pkm_array[10] = min(999,max(1,math.floor(boosts_dict[pkm_array[16]]*pkm_array[9])))
                    opp_view_pkm_array[16] = pkm_array[16]
                    opp_view_pkm_array[10] = min(999,max(1,math.floor(boosts_dict[opp_view_pkm_array[16]]*opp_view_pkm_array[9])))
                    if opp_pkm_array[3] == 1:
                        opp_pkm_array[13]//=2
                        play_view_opp_array[13]//=2
                    elif opp_pkm_array[3] == 4:
                        opp_pkm_array[10]//=4
                        play_view_opp_array[10]//=4
                elif 'spd' in move_boosts:
                    boost_num = move_boosts['spd']
                    if boost_num>0 and pkm_array == 13:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    elif boost_num< 0 and pkm_array == 1:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    
                    pkm_array[17] = boost_num+pkm_array[17]
                    opp_view_pkm_array[17] = pkm_array[17]
                    if opp_pkm_array[3] == 1:
                        opp_pkm_array[13]//=2
                        play_view_opp_array[13]//=2
                    elif opp_pkm_array[3] == 4:
                        opp_pkm_array[10]//=4
                        play_view_opp_array[10]//=4

                elif 'atk' in move_boosts:
                    
                    boost_num = move_boosts['atk']
                    if boost_num>0 and pkm_array == 13:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    elif boost_num< 0 and pkm_array == 1:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    
                    pkm_array[18] = boost_num+pkm_array[18]
                    pkm_array[13] = min(999,max(1,math.floor(boosts_dict[pkm_array[18]]*pkm_array[12])))
                    opp_view_pkm_array[18] = pkm_array[18]
                    opp_view_pkm_array[13] = min(999,max(1,math.floor(boosts_dict[opp_view_pkm_array[18]]*opp_view_pkm_array[12])))
                    if opp_pkm_array[3] == 1:
                        opp_pkm_array[13]//=2
                        play_view_opp_array[13]//=2
                    elif opp_pkm_array[3] == 4:
                        opp_pkm_array[10]//=4
                        play_view_opp_array[10]//=4
                elif 'def' in move_boosts:
                    
                    boost_num = move_boosts['def']
                    if boost_num>0 and pkm_array == 13:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    elif boost_num< 0 and pkm_array == 1:
                        return (player_fnt,opp_fnt,flinch_status,clamp_status)
                    
                    pkm_array[19] = boost_num+pkm_array[19]
                    opp_view_pkm_array[19] = pkm_array[19]
                    if opp_pkm_array[3] == 1:
                        opp_pkm_array[13]//=2
                        play_view_opp_array[13]//=2
                    elif opp_pkm_array[3] == 4:
                        opp_pkm_array[10]//=4
                        play_view_opp_array[10]//=4
                return (player_fnt,opp_fnt,flinch_status,clamp_status)                   


                
        
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
            active_list = [p1_active_array,p2_active_array]
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
            p1_move_num = p1_active_array[22+ 2*p1_action]
            p1_move_name = move_dict[p1_move_num]
            p2_move_num = p2_active_array[22 + 2*p2_action]
            p2_move_name = move_dict[p2_move_num]

            if p1_move_name == 'quickattack' and p2_move_name != 'quickattack':
                first_mover = 'player1'
            elif p1_move_name != 'quickattack' and p2_move_name == 'quickattack':
                first_mover = 'player2'
            
            if p1_move_name == 'counter' and p2_move_name != 'counter':
                first_mover == 'player2'
            elif p1_move_name != 'counter' and p2_move_name == 'counter':
                first_mover == 'player1'

            if first_mover == 'player1':
                outcome_1 = self.reg_move(0,active_list,move=p1_move_name,clamp_flag=False)
                play1_fnt = outcome_1[0] 
                play2_fnt = outcome_1[1]
                p2_flinch_status = outcome_1[2]
                p2_clamp_status = outcome_1[3]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        high_values = np.array([255,1,1,31,15,15,63,1023,1023,1023,1023,1023,1023,1023,1023,127,15,15,15,15,3,255,127,255,127,255,127,255,127,1023,511])
        high_matrix = np.tile(high_values, (12, 32)) 
        observation_spaces = {'p1':Box(
                    low=0,  
                    high=high_matrix,  
                    shape=(12, 32), 
                    dtype=np.uint16
                ), 'p2':Box(
                    low=0,  
                    high=high_matrix,  
                    shape=(12, 32), 
                    dtype=np.uint16
                )}
        return observation_spaces
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(11)
# from pettingzoo.test import parallel_api_test
# env = Pokebot_Gen1()
# parallel_api_test(env,num_cycles=1)