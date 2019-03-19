import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import sys

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_good_agents = 10
        num_adversaries = 10
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 8 #88
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.04 if agent.adversary else 0.04
            agent.accel = 3.0 if agent.adversary else 3.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            # agent.max_speed = 1.0 if agent.adversary else 1.0
            #! fast agents shows wrong movements
            if i%2 ==0 :
                agent.max_speed = 1.0/5  if agent.adversary else 1.0/5 
            else:
                agent.max_speed = 1.0/5 - 0.1 if agent.adversary else 1.0/5 - 0.1
        #! add landmarks vertical and horizontal
        world.landmarks = [Landmark('ver') if i < num_landmarks/2 else Landmark('hor') for i in range(num_landmarks)]
        # print([(i, lan.pos) for i,lan in enumerate(world.landmarks) ])
        # sys.exit()

        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.05
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            if agent.adversary: #! position of the agents 
                agent.state.p_pos = np.random.uniform(-0.19, +0.19, world.dim_p) + np.array([0,0.7])
            else:
                agent.state.p_pos = np.random.uniform(-0.19, +0.19, world.dim_p) + np.array([0.7,0])
 
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks): #! position of the walls
            
            
        
            if i > 5 : #horizontal
                landmark.state.p_pos = np.array([0.635,((i-5)-1.5)/1.5])
            elif i > 3 : #horizontal
                landmark.state.p_pos = np.array([-0.635,((i-3)-1.5)/1.5])
           
            elif i > 1 : #vertical
                landmark.state.p_pos = np.array([((i-1)-1.5)/1.5, 0.635])
            elif i > -1 : #vertical
                landmark.state.p_pos = np.array([((i+1)-1.5)/1.5,-0.635])

            # if i>=2: #horizontal
            #     landmark.state.p_pos = np.array([0.7,-0.5])
            # if i>=0: #vertical 
            #     landmark.state.p_pos = np.array([((i-1)+0.5)*1.2,0.5])
            # landmark.state.p_pos = np.array([-0.6,0])
            # landmark.state.p_pos = np.array([-0.6,0])

            # landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)

            # Position of the cross roads
            # if i >= 70:
            #     landmark.state.p_pos = np.array([-0.2,(i-68)*(2/20)])
            # elif i >= 60:
            #     landmark.state.p_pos = np.array([-0.2,(i-71)*(2/20)])

            # elif i >= 50:
            #     landmark.state.p_pos = np.array([0.2,(i-48)*(2/20)])
            # elif i >= 40:
            #     landmark.state.p_pos = np.array([0.2,(i-51)*(2/20)])

            
            # elif i >= 30:
            #     landmark.state.p_pos = np.array([(i-28)*(2/20),-0.2])
            # elif i >= 20:
            #     landmark.state.p_pos = np.array([(i-31)*(2/20),-0.2])

            # elif i >= 10:
            #     landmark.state.p_pos = np.array([(i-8)*(2/20),0.2])
            # elif i >= 0:
            #     landmark.state.p_pos = np.array([(i-11)*(2/20),0.2])


            


            #     landmark.state.p_pos = np.array([(i-39)*(2/20)-1,0.4])
            # elif i > 20:
            #     landmark.state.p_pos = np.array([-0.4,(i-19)*(2/20)-1])
            # else:
            #     landmark.state.p_pos = np.array([0.4,(i)*(2/20)-1])

            landmark.state.p_vel = np.zeros(world.dim_p)
        
        # for i, landmark in enumerate(world.landmarks):
        #     if not landmark.boundary:
        #         # landmark.state.p_pos = [-0.8+i*(1.6/len(world.landmarks)) ,0.8]
        #         landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
        #         landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns number of collisions for adversary agent
        if agent.adversary:
            collisions = 0
            for ga in self.good_agents(world):
                if self.is_collision(ga, agent):
                    collisions += 1
            return collisions

        if not agent.adversary:
            collisions = 0
            for adv in self.adversaries(world):
                if self.is_collision(adv, agent):
                    collisions += 1
            return collisions
        else:
            return -1


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def bound(self,x):
        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        if x < 0.9:
            return 0
        
        return min(np.exp(2 * x - 2), 10)

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False #!False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10
        #! punishment 
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= self.bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False #!False 
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        #! punishment                
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= self.bound(x)
            
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # if not other.adversary:
            other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    