import random 
import pandas as pd 
import pickle
from poke_env.environment.move import Move
with open('pkm_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

pokemon_list = loaded_data['pokemon_list']
pkm_mvs_dict = loaded_data['pkm_mvs_dict']
stats = pd.read_csv('pokemon-bst-totals.csv')
stats['Name'] = stats['Name'].apply(str.lower)
stats.set_index('Name',inplace=True)
class Pokemon_s():

    def __init__(self,species:str,stats:dict[str:int],moves:list[str],level:int):
        self._species = species
        self._active = 0
        self._alt_status = 0
        self._toxic_counter = 1
        self._status = 0
        self._hp = stats['hp']
        self._current_hp = stats['hp']
        self._defense = stats['def']
        self._speed = stats['spe']
        self._special = stats['sp']
        self._attack = stats['atk']
        self._level = level
        move_dict = {}
            

        for i in range(len(moves)):
            mv = Move(move_id=moves[i],gen=1)
            key_name = 'Move' + ' ' + str(i+1)
            move_dict[key_name] = mv
        self._move_1 = move_dict.get('Move 1','nomove')
        self._move_2 = move_dict.get('Move 2','nomove')
        self._move_3 = move_dict.get('Move 3','nomove')
        self._move_4 = move_dict.get('Move 4','nomove')
        
    @property
    def species(self):
        return self._species
    
    @property
    def active(self):
        if self._active == 0:
            return False
        elif self._active ==1:
            return True
    @active.setter
    def active(self,status:bool):
        if status:
            self._active = 1
        else:
            self._active = 0
    @property
    def alt_status(self):
        return self._alt_status
    @alt_status.setter
    def alt_status(self,status:int):
        self._alt_status = status
    @property
    def toxic_counter(self):
        return self._toxic_counter
    @toxic_counter.setter
    def toxic_counter(self,status:int):
        self._toxic_counter = status

    @property
    def hp(self):
        return self._hp
    @property
    def defense(self):
        return self._defense
    
    @property
    def attack(self):
        return self._attack
    @property
    def speed(self):
        return self._speed
    
    @property
    def special(self):
        return self._special
    
    @property
    def move_1(self):
        return self._move_1
    @property
    def move_2(self):
        return self._move_2
    
    @property
    def move_3(self):
        return self._move_3
    @property
    def move_4(self):
        return self._move_4
banned_moves = ['guillotine','whirlwind','stomp','fly','jumpkick','rollingkick',
                'thrash','roar','disable','solarbeam','mist','petaldance'
                ,'haze','focusenergy','bide','metronome','mirrormove','boneclub',
                'highjumpkick','transform','splash','']
def team_generator(seed_num=None):

    random.seed(seed_num)
    mon_nums = []
    while len(mon_nums)<6:
        num = random.randint(1,149)

        if num not in mon_nums:
            mon_nums.append(num)

    mon_list = [pokemon_list[x] for x in mon_nums]
    mon_moves = []
    mon_stats = []
    mon_levels = [100,100,100,100,100,100]
    for mon in mon_list:
        valid_move_list = [x for x in pkm_mvs_dict[mon] if x not in banned_moves] 
        mvs = random.sample(pkm_mvs_dict[mon],min(4,len(pkm_mvs_dict[mon])))

        stats_dict = {'atk': stats.loc[mon]['Attack_Total'] , 'def': stats.loc[mon]['Defense_Total'] , 'hp': stats.loc[mon]['HP_Total'],
                        'sp': stats.loc[mon]['Special_Total'], 'spe': stats.loc[mon]['Speed_Total']}
        mon_moves.append(mvs)

        mon_stats.append(stats_dict)
    pkm_team = []

    for i in range(len(mon_list)):

        pokemon = Pokemon_s(mon_list[i],mon_stats[i],mon_moves[i],mon_levels[i])


        pkm_team.append(pokemon)
    
    return pkm_team