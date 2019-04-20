import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import sys
from grid_to_map import *

#* map 

grid1 = np.array([
    [0,0,0,0,1,1,1,1,0,0,0,0],
	[0,0,0,0,1,0,0,1,0,0,0,0],
	[0,0,0,0,1,0,0,1,0,0,0,0],
	[0,0,0,0,1,0,0,1,0,0,0,0],
	[1,1,1,1,1,0,0,1,1,1,1,1],
	[1,0,0,0,0,0,0,0,0,0,0,1],
	[1,0,0,0,0,0,0,0,0,0,0,1],
	[1,1,1,1,1,0,0,1,1,1,1,1],
	[0,0,0,0,1,0,0,1,0,0,0,0],
	[0,0,0,0,1,0,0,1,0,0,0,0],
	[0,0,0,0,1,0,0,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,0,0,0,0]

])

grid2 = np.array([
	[0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,1,1,0,0,0,0],
	[0,0,0,1,1,1,1,0,0,0],
	[0,0,1,1,0,0,1,1,0,0],
	[0,0,1,1,0,0,1,1,0,0],
	[0,1,1,1,1,1,1,1,1,0],
	[0,1,1,1,1,1,1,1,1,0],
	[1,1,1,0,0,0,0,1,1,1],
	[1,1,0,0,0,0,0,0,1,1],
	[1,0,0,0,0,0,0,0,0,1]
]) 
grid3 = np.array([
	[0,0,0,0,0,0,0,0,0,0],
	[0,0,1,1,0,0,0,0,1,1],
	[0,0,1,1,0,0,0,1,1,0],
	[0,0,1,1,0,0,1,1,0,0],
	[0,0,1,1,1,1,1,0,0,0],
	[0,0,1,1,1,1,1,0,0,0],
	[0,0,1,1,0,0,1,1,0,0],
	[0,0,1,1,0,0,0,1,1,0],
	[0,0,1,1,0,0,0,0,1,1],
	[0,0,0,0,0,0,0,0,0,0]
]) 
class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        num_of_group1 = 10
        num_of_group2 = 10
        num_agents = num_of_group2 + num_of_group1
        num_landmarks = getnumberofwall(grid1)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.group2 = True if i < num_of_group2 else False
            agent.size = 0.04 if agent.group2 else 0.04
            agent.accel = 3.0 if agent.group2 else 3.0
            #agent.accel = 20.0 if agent.group2 else 25.0
            # agent.max_speed = 1.0 if agent.group2 else 1.0
            #! fast agents shows wrong movements
            if i%2 == 0 :
                agent.max_speed = 1.0/5  if agent.group2 else 1.0/5 
            else: #! for now, every agent has the same speed
                agent.max_speed = 1.0/5 if agent.group2 else 1.0/5
        #! add landmarks vertical and horizontal
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        # print([(i, lan.pos) for i,lan in enumerate(world.landmarks) ])
        # sys.exit()

        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0 #0.05
            landmark.boundary = False
            landmark.shape = (0.2, 0.2)
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.group2 else np.array([0.85, 0.35, 0.35])
            #! isDone is reset back to false and make it collidable
            agent.isDone = False
            agent.collide = True
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            if agent.group2: #! position of the agents
                # agent.state.p_pos = np.array([-1,0]) + agent.size
                agent.state.p_pos = np.random.uniform(-0.19, +0.19, world.dim_p) + np.array([0,0.7])
                agent.destination = [-0.2,0.2,-1,-0.8] #! destination of the agents in group2
            else: #group1
                agent.state.p_pos = np.random.uniform(-0.19, +0.19, world.dim_p) + np.array([0.7,0])
                agent.destination = [-1,-0.8,-0.2,0.2] #! destination of the agents in group1
 
            agent.state.p_vel = np.zeros(world.dim_p)
        
        for i, landmark in enumerate(world.landmarks): #! position of the walls
            landmark.state.p_pos = np.array(grid2pos(grid1, 0.2))[i]
            landmark.state.p_vel = np.zeros(world.dim_p)
   
    #! Can use collision detection in step
    def benchmark_data(self, agent, world):
        # returns number of collisions        
        if agent.isCollided:
            return 1
        return 0


    # return all agents that are group1
    def group1_agents(self, world):
        return [agent for agent in world.agents if not agent.group2]

    # return all agents that are group2
    def group2_agents(self, world):
        return [agent for agent in world.agents if agent.group2]


    def reward(self, agent, world):
        # Reward for an agent: 2 - current distance between agent and its destination 
        if agent.isDone:
            return 2
        delta = ((agent.state.p_pos[0] - (agent.destination[0] + agent.destination[1])/2)**2 +\
                (agent.state.p_pos[1] - (agent.destination[2] + agent.destination[3])/2)**2)**1/2
        
        reward = 2 - delta

        if agent.isCollided:
            reward -= 1

        return reward

    

    def observation(self, agent, world):
        # agent-lanmark
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                #position of lanmark relative to the agent
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # agent-agent
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue #no interaction with itself
            #position of other agent relative to the agent
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            #velocity of other agent relative to the agent
            other_vel.append(other.state.p_vel - agent.state.p_vel)
        
        # [agent's velocity(2d vector) + agent's position(2d vector) +
        # landmark's relative position(k*2d vector)
        # other agent's relative position((n-1)*2d vector) +
        # other agent's relative velocity((n-1)*2d vector)) ]
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    