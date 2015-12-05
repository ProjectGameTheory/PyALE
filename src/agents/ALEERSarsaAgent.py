# -*- coding: utf-8 -*-


from rlglue.agent import AgentLoader as AgentLoader



import argparse

from util.ALEFeatures import BasicALEFeatures,RAMALEFeatures
from agents.ALESarsaAgent import ALESarsaAgent


import numpy as np

class ALEERSarsaAgent(ALESarsaAgent):
    
    def __init__(self,sample_size=1000, nb_times=2, **kwargs):
        super(ALEERSarsaAgent,self).__init__(**kwargs)
        
        # only last 'sample_size' samples will be saved
        self.max_size = sample_size
        self.current_size = 0        
        self.samples = [None]*self.max_size
        self.nb_times = nb_times

    def collectSample(self, reward, phi_ns):
        # where to save in array
        index = self.current_size % self.max_size
        self.samples[index] = [self.phi, self.a, phi_ns, reward]
        self.current_size += 1
        
    def getRandomSample(self):
        # pick random sample from saved sample
        # return array [state_k, action_k, state_k+1, reward]
        size = min(self.current_size, self.max_size)
        return self.samples[np.random.randint(0, size)]

    def update(self, sample):
        phi, act, phi_ns, reward = sample
        n_rew = self.normalize_reward(reward)
        self.update_trace(phi,act)
        delta = n_rew - self.get_value(phi,act,self.sparse)
        a_ns = None
        if not (phi_ns is None):
            ns_values = self.get_all_values(phi_ns,self.sparse)
            a_ns = self.select_action(ns_values)
            delta += self.gamma*ns_values[a_ns]
        #normalize alpha with nr of active features
        alpha = self.alpha / float(np.sum(self.phi!=0.))
        self.theta+= alpha*delta*self.trace
        return a_ns  #a_ns is action index (not action value)
        
    def step(self,reward,phi_ns = None):
        a_ns = None
        if not (phi_ns is None):
            ns_values = self.get_all_values(phi_ns,self.sparse)
            a_ns = self.select_action(ns_values)
        return a_ns  #a_ns is action index (not action value)

    def agent_step(self,reward, observation):
        #super(ALESarsaAgent,self).agent_step(reward, observation)
        phi_ns = self.get_phi(observation)
        self.collectSample(reward, phi_ns)
        a_ns = self.step(reward,phi_ns)
        #log state data
        self.phi = phi_ns
        self.a = a_ns 
        
        return self.create_action(self.actions[a_ns])#create RLGLUE action
        
        
             
    def agent_end(self,reward):
        super(ALESarsaAgent,self).agent_end(reward)
        size = min(self.current_size, self.max_size)
        for i in range(self.nb_times*size):
            sample = self.getRandomSample()
            self.update(sample)
        self.step(reward)
        
class BasicALEERSarsaAgent(ALEERSarsaAgent):
    def __init__(self,bg_file='../data/space_invaders/background.pkl',**kwargs):
        super(BasicALEERSarsaAgent,self).__init__(**kwargs)
        self.background = bg_file
        
    def create_projector(self):
        return BasicALEFeatures(num_tiles=np.array([14,16]),
            background_file =  self.background,secam=True )
 
    def get_data(self,obs):
        return self.get_frame_data(obs)

    
class RAMALEERSarsaAgent(ALEERSarsaAgent):
    def create_projector(self):
        return RAMALEFeatures()
        
    def get_data(self,obs):
        return self.get_ram_data(obs)

        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run Sarsa Agent')
    parser.add_argument('--id', metavar='I', type=int, help='agent id',
                        default=0)
    parser.add_argument('--gamma', metavar='G', type=float, default=0.999,
                    help='discount factor')
    parser.add_argument('--alpha', metavar='A', type=float, default=0.5,
                    help='learning rate')
    parser.add_argument('--lambda_', metavar='L', type=float, default=0.9,
                    help='trace decay')
    parser.add_argument('--eps', metavar='E', type=float, default=0.05,
                    help='exploration rate')
    parser.add_argument('--savepath', metavar='P', type=str, default='.',
                    help='save path')  
    parser.add_argument('--features', metavar='F', type=str, default='RAM',
                    help='features to use: RAM or BASIC')
                    
    parser.add_argument('--actions', metavar='C',type=int, default=None, 
                        nargs='*',help='list of allowed actions')

    parser.add_argument('--sample_size', metavar='S',type=int, default=1000, 
                        nargs='*',help='size of sample database')

    parser.add_argument('--nb_times', metavar='N',type=int, default=2, 
                        nargs='*',help='nb loops over sample db for one update')

    args = parser.parse_args()
    
    act = None
    if not (args.actions is None):
        act = np.array(args.actions)

    if args.features == 'RAM':
        AgentLoader.loadAgent(RAMALEERSarsaAgent(agent_id=args.id,
                                     alpha =args.alpha,
                                     lambda_=args.lambda_,
                                     eps =args.eps,
                                     gamma=args.gamma, 
                                     save_path=args.savepath,
                                     actions = act,
                                     sample_size=args.sample_size,
                                     nb_times=args.nb_times))
    elif args.features == 'BASIC':
        AgentLoader.loadAgent(BasicALEERSarsaAgent(agent_id=args.id,
                                     alpha =args.alpha,
                                     lambda_=args.lambda_,
                                     eps =args.eps,
                                     gamma=args.gamma, 
                                     save_path=args.savepath,
                                     actions = act,
                                     sample_size=args.sample_size,
                                     nb_times=args.nb_times))
    else:
        print 'unknown feature type'
    
        
