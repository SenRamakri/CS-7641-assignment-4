import sys

import numpy as np
from gym import utils
from gym.envs.toy_text import discrete
from six import StringIO

VALUELINE = "SNNRNRNRNRNNRNNRNNRT"

class ValueLineEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, is_slippery=True):
        desc = VALUELINE
        self.desc = desc = np.asarray(desc, dtype='c')
        self.is_slippery = is_slippery
        self.state = 0
        
        nA = 5 #-2 -1 0 +1 +2
        nS = len(VALUELINE)
        self.max_state = nS-1
        self.min_state = 0
        self.ncol = nS
        self.nrow = 1

        isd = np.array(desc == b'S').astype('float64').ravel()
        #print("isd:%s"%isd)
        isd /= isd.sum()
        #print("isd2:%s"%isd)

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for s_i in range(nS):
            for a in range(5):
                li = P[s_i][a]
                a = a-2
                new_state = s_i + a
                if(new_state > self.max_state):
                    new_state = self.max_state
                if(new_state < 0):
                    new_state = 0
                state_desc = VALUELINE[new_state]
                #print("s_i=%s a=%s new_state=%s state_desc:%s"%(str(s_i),str(a),new_state,state_desc))   
                if (state_desc == 'R' or state_desc == 'S'):
                    new_state = 0
                    li.append((1.0, new_state, -10, False)) #prob, next_state, reward, done
                if (state_desc == 'N'):
                    li.append((1.0, new_state, -1, False)) #prob, next_state, reward, done
                if (state_desc == 'T'):
                    li.append((1.0, new_state, 10, True)) #prob, next_state, reward, done

        #print("P=%s"%str(P))
        super(ValueLineEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile

    def colors(self):
        return {
            b'S': 'green',
            b'R': 'red',
            b'N': 'lightgreen',
            b'T': 'gold'
        }
    
    def directions(self):
        return {
            4: '+2',
            3: '+1',
            2: '0',
            1: '-1',
            0: '-2'
        }

    def new_instance(self):
        return ValueLineEnv(desc=self.desc, is_slippery=self.is_slippery)
