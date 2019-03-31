import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random 
class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        num_good_agents = 5
        num_adversaries = 5
        num_landmarks = 40
        num_agents = num_adversaries + num_good_agents
        world.agents = [Agent() for i in range(num_agents)]
        # add agents
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.075
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            # agent.max_speed = 1.0/5 if agent.adversary else 1.3/5

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.05
            landmark.shape = (0.2, 0.8)
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
            landmark.color = np.array([0.75,0.75,0.75])
        # set random initial states
        for agent in world.agents:
            # agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_pos = [0.5, 0.6]
            # [0.8+random.uniform(-0.1,0.1), 0.5+random.uniform(-0.1,0.1)] if not agent.adversary else [0.2+random.uniform(-0.1,0.1), 0.5+random.uniform(-0.1,0.1)]
            agent.state.p_vel = np.zeros(world.dim_p)
            # agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            if i < 19:
                landmark.state.p_pos = np.array([i*(2/20)-1,0.8])
            else:
                landmark.state.p_pos = np.array([(i-20)*(2/20)-1,0.2])

            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return 10

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # entity_pos = []
        # for entity in world.landmarks:
        #     if not entity.boundary:
        #         entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # # communication of all other agents
        # comm = []
        # other_pos = []
        # other_vel = []
        # for other in world.agents:
        #     if other is agent: continue
        #     comm.append(other.state.c)
        #     other_pos.append(list(np.array(other.state.p_pos) - np.array(agent.state.p_pos)))
        #     # if not other.adversary:
        #     other_vel.append(other.state.p_vel)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
        return np.random.ranf((2+9*2*2))