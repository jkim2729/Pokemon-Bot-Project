import math
import random
# Precompute the dmg_list outside the function as it does not change
DAMAGE_LIST = [217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]

def damage(level, crit , power , atk, dfs, atk_mod = 1, dfs_mod = 1, stab = 1, type_mult = 1): 
    """
    Function to calculate damage
    level - attacker pokemon level
    crit - True or False depending of if move is a critical hit
    power - base power of move
    atk - attack stat of pokemon based on move type (special/physical)
    dfs - defense stat of pokemon based on move type (special/physical)
    stab - 1.5 if stab multiplier, 1 otherwise
    type_mult - based on type charts can range from 0-4

    """


    # Precompute common values
    lvl_value = (2 + (4 * level) // 5) if crit else (2 + (2 * level) // 5)
    max_atk_dfs = max(atk, dfs)
    
    # Check if atk or dfs exceeds 255 and adjust values accordingly
    if max_atk_dfs > 255:
        # Scaling for high attack or defense values
        at = (atk // 4) if crit else ((atk_mod * atk) // 4)
        df = (dfs // 4) if crit else ((dfs_mod * dfs) // 4)
    else:
        # No scaling for normal values
        at = atk if crit else (atk_mod * atk)
        df = dfs if crit else (dfs_mod * dfs)

    # Calculate damage
    dmg_1 = ((lvl_value * power * at) // df) // 50 + 2
    final_dmg_value = math.floor(math.floor(dmg_1 * stab) * type_mult)

    # If final damage value is 1, return a list of 1s
    if final_dmg_value == 1:
        return [1] * 39

    # Otherwise, return a scaled list of damage values
    return [(final_dmg_value * x) // 255 for x in DAMAGE_LIST]

def select_dmg_value(level,speed,power,atk,dfs,atk_mod = 1,dfs_mod = 1,stab = 1,type_mult = 1,crit_status ='na'): #base speed
    if crit_status == 'na':
        crit_rate = max(255/256,(speed/512))
        rand_num = random.random()
        if rand_num<crit_rate:
            crit_val = True
        else:
            crit_val = False
    else:
        crit_val = crit_status
    dmg_values = damage(level,crit_val,power,atk,dfs,atk_mod,dfs_mod,stab,type_mult)
    final_val = random.choice(dmg_values)

    return final_val

def confusion_dmg(level,atk,dfs,atk_mod,dfs_mod):

    dmg_val = max(damage(level=level,crit=False,power=40,atk=atk,dfs=dfs,atk_mod=atk_mod,dfs_mod=dfs_mod,stab=1,type_mult=1))
    return dmg_val
