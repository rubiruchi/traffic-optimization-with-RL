import numpy as np
import math
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
        num_of_group1 = 5
        num_of_group2 = 5
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
        '''
        Agent's observation: position + destination + velocity + it's 4 sensor measurements
        '''
        other_agent_positions = [a.state.p_pos for a in world.give_agents if a is not self]
        wall_positions = [l.state.p_pos for l in world.give_landmarks]
        wall_length = world.give_landmarks[0].shape[0]
        agent_position = agent.state.p_pos
        agent_radius = agent.size 
        max_sensor_dist = agent.s_dist

        res = getsensormeasurements(other_agent_positions, wall_positions, wall_length, agent_position, agent_radius, max_sensor_dist)
        # print('-'*50)
        # print(np.array(res))
        res = np.array(res)
        des = np.array([np.mean(agent.destination[:2]), np.mean(agent.destination[2:])])
        res_ = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [des] + [res])
        return res_

#! Helper functions for sensor measurements
def getline(wall_position, wall_length):
    # line order: left,right,down,up
    # line format [x,y] as either fixed x or fixed y
    line1 = np.array([wall_position[0]-wall_length/2, None], dtype='double')
    line2 = np.array([wall_position[0]+wall_length/2, None], dtype='double')
    line3 = np.array([None, wall_position[1]-wall_length/2], dtype='double')
    line4 = np.array([None, wall_position[1]+wall_length/2], dtype='double')

    return([line1, line2, line3, line4])

def getsensormeasurements_(wall_position, wall_length, agent_position, agent_radius, max_sensor_dist):
    '''
    Returns an array of sensor measurements (distances) in form of: [left,right,down,up]
    Looks for walls
    '''
    measurements = [max_sensor_dist]*4
    # calculations start
    [line1, line2, line3, line4] = getline(wall_position, wall_length)

    # left sensor
    if (wall_position[1] - agent_position[1])**2 < (wall_length/2)**2:
        dist1 = agent_position[0] - line2[0]
        if dist1 > 0:
            if dist1 < max_sensor_dist:
                measurements[0] = dist1

    # right sensor
    if (wall_position[1] - agent_position[1])**2 < (wall_length/2)**2:
        dist2 = line1[0] - agent_position[0]
        if dist2 > 0:
            if dist2 < max_sensor_dist:
                measurements[1] = dist2

    # down sensor
    if (wall_position[0] - agent_position[0])**2 < (wall_length/2)**2:
        dist3 = agent_position[1] - line4[1]
        if dist3 > 0:
            if dist3 < max_sensor_dist:
                measurements[2] = dist3

    # up sensor
    if (wall_position[0] - agent_position[0])**2 < (wall_length/2)**2:
        dist4 = line3[1] - agent_position[1]
        if dist4 > 0:
            if dist4 < max_sensor_dist:
                measurements[3] = dist4

    # calculations end
    return measurements

def getsensormeasurements_2(other_agent_pos, agent_position, agent_radius, max_sensor_dist):
    '''
    Returns an array of sensor measurements (distances) in form of: [left,right,down,up]
    Looks for other agents
    '''
    measurements = [max_sensor_dist]*4
    
    # left sensor
    if (other_agent_pos[1] - agent_position[1])**2 < (agent_radius)**2:
        c =  agent_position - other_agent_pos
        b = c[0]
        a = c[1]
        dist1 = b - math.sqrt(agent_radius**2 - a**2) 
        if dist1 > 0:
            if dist1 < max_sensor_dist:
                measurements[0] = dist1

    # right sensor
    if (other_agent_pos[1] - agent_position[1])**2 < (agent_radius)**2:
        c =  other_agent_pos - agent_position
        b = c[0]
        a = c[1]
        dist2 = b - math.sqrt(agent_radius**2 - a**2) 
        if dist2 > 0:
            if dist2 < max_sensor_dist:
                measurements[1] = dist2

    # down sensor
    if (other_agent_pos[0] - agent_position[0])**2 < (agent_radius)**2:
        c =  agent_position - other_agent_pos
        b = c[1]
        a = c[0]
        dist3 = b - math.sqrt(agent_radius**2 - a**2) 
        if dist3 > 0:
            if dist3 < max_sensor_dist:
                measurements[2] = dist3

    # up sensor
    if (other_agent_pos[0] - agent_position[0])**2 < (agent_radius)**2:
        c =  other_agent_pos - agent_position
        b = c[1]
        a = c[0]
        dist4 = b - math.sqrt(agent_radius**2 - a**2) 
        if dist4 > 0:
            if dist4 < max_sensor_dist:
                measurements[3] = dist4
                
    return measurements

def getsensormeasurements(other_agent_positions, wall_positions, wall_length, agent_position, agent_radius, max_sensor_dist):
    '''
    Returns an array of sensor measurements (distances) in form of: [left,right,down,up]
    '''
    measurements = [max_sensor_dist]*4
    #calculations start
    
    meas1 = np.array([getsensormeasurements_(w_pos,wall_length,agent_position,agent_radius,max_sensor_dist) for w_pos in wall_positions]) 
    meas1_ = meas1.min(axis=0)
    
    meas2 = np.array([getsensormeasurements_2(o_a_pos, agent_position, agent_radius, max_sensor_dist) for o_a_pos in other_agent_positions])
    meas2_ = meas2.min(axis=0)
    
    # pprint.pprint(meas1)
    # print('-'*30)
    # pprint.pprint(meas2)
    
    #calculations end
    return np.vstack((meas1_,meas2_)).min(axis=0)

    