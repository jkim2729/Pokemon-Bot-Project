// calc_test.js
import { calculate, Generations, Pokemon, Move } from '@smogon/calc';

const gen = Generations.get(1); // Generation 5
const result = calculate(
  gen,
  new Pokemon(gen, 'Snorlax', {
    evs: {hp: 252, atk: 252, def: 252, spa: 252 , spd: 252, spe: 252},
    dvs: {hp: 15, atk: 15, def: 15, spa: 15 , spd: 15, spe: 15},
  }),
  new Pokemon(gen, 'Chansey', {
    evs: {hp: 252, atk: 252, def: 252, spa: 252 , spd: 252, spe: 252},
    dvs: {hp: 15, atk: 15, def: 15, spa: 15 , spd: 15, spe: 15},
  }),
  new Move(gen, 'Body Slam')
);

// Output the result or relevant data
console.log(result);
