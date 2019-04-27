'''
Entity state
Agent state
Action
Entity
Landmark
Agent
'''

import numpy as np
import sys
import pprint
# physical/external base state of all entites


class EntityState(object):
    '''
    Consists of position and velocity of the entity
    '''

    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


class AgentState(EntityState):
    '''
    Same as entity state
    '''

    def __init__(self):
        super(AgentState, self).__init__()

#! action of the agent


class Action(object):
    '''
    Physical action
    '''

    def __init__(self):
        # physical action
        self.u = None

#! properties and state of physical world entity


class Entity(object):
    '''
    Entity object has a name, size, movable, collide, density, color, 
    max speed, acceleration, entity state and inital mass
    '''

    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities


class Landmark(Entity):
    '''
    Landmarks have an argument about their position hor or vec
    '''

    def __init__(self, shape=[0.2, 0.2]):
        super(Landmark, self).__init__()
        # self.pos = pos
        self.shape = shape


def isIn(agentpos, area):
    if(area[0] <= agentpos[0] and
       agentpos[0] <= area[1] and
       area[2] <= agentpos[1] and
       agentpos[1] <= area[3]):
        return True
    return False

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


def getsensormeasurements(wall_positions, wall_length, agent_position, agent_radius, max_sensor_dist):
    '''
    Returns an array of sensor measurements (distances) in form of: [left,right,down,up]
    '''
    measurements = [max_sensor_dist]*4
    # calculations start
    meas = np.array([getsensormeasurements_(w_pos, wall_length, agent_position,
                                            agent_radius, max_sensor_dist) for w_pos in wall_positions])
    # calculations end
    return meas.min(axis=0)

#! properties of agent entities

class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # is it collided
        self.isCollided = False
        # did it reached its destination
        self.isDone = False
        # isDone previous step
        self.isDone_ = False
        # did it collided before destination
        self.isWreck = False
        # isWreck previous step
        self.isWreck_ = False
        # it's destination
        self.destination = [1]*4  # [-0.2,0.2,-0.2,0.2]
        # it's max sensor distances
        self.s_dist = 0.1

    # check if the agent reached it's destination
    def isReached(self):
        if isIn(self.state.p_pos, self.destination):
            # print('It is in the area!')
            self.isDone = True
            # self.movable = False
            self.collide = False

  
    # @property
    #! return it's sensors readings
    def sensors(self, world):

        wall_positions = [l.state.p_pos for l in world.give_landmarks]
        wall_length = world.give_landmarks[0].shape[0]
        agent_position = self.state.p_pos
        agent_radius = self.size 
        max_sensor_dist = self.s_dist

        res = getsensormeasurements(wall_positions, wall_length, agent_position, agent_radius, max_sensor_dist)

        sensors = [[- res[0], 0],
                   [+ res[1], 0],
                   [0, - res[2]],
                   [0, + res[3]]]

        pprint.pprint(sensors)           
        return sensors


# multi-agent world
class World(object):
    def __init__(self):
        self.timestep = 0
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        #! simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    @property
    def give_agents(self):
        return self.agents

    @property
    def give_landmarks(self):
        return self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        #! set collision to false
        for agent in self.give_agents:
            agent.isCollided = False
            agent.isWreck_ = agent.isWreck #previous isWreck
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)  #current isWreck

        # update agent state(isDone,collide)
        for agent in self.agents:
            agent.isDone_ = agent.isDone #previous isDone
            if not agent.isDone:
                agent.isReached() #current isDone

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(
                    *agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        a = [type(entity) for entity in self.entities]

        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if(b <= a):
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None):
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if(f_b is not None):
                    if(p_force[b] is None):
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                #! velocity calculation
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    #! Collision was wrong for square objects, reformulated it
    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        # get collision
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself

        # * if entity_a is a Agent & entity_b is a Landmark
        if (isinstance(entity_a, Agent) and isinstance(entity_b, Landmark)):

            dumping = 0.001
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            if(np.abs(delta_pos[0]) - dumping <= entity_b.shape[0]/2 + entity_a.size):
                if(np.abs(delta_pos[1]) - dumping <= entity_b.shape[1]/2 + entity_a.size):
                    # * collison
                    entity_a.isCollided = True
                    entity_a.isWreck = True

                    # print('Collision with wall')
                    dist = np.sqrt(np.sum(np.square(delta_pos)))

                    upperline = entity_b.state.p_pos[1] + entity_b.shape[1]/2
                    lowerline = entity_b.state.p_pos[1] - entity_b.shape[1]/2
                    rightline = entity_b.state.p_pos[0] + entity_b.shape[0]/2
                    leftline = entity_b.state.p_pos[0] - entity_b.shape[0]/2

                    # * if horizontal collision
                    if(leftline <= entity_a.state.p_pos[0] and
                            rightline >= entity_a.state.p_pos[0]):

                        # * apply horizontal force
                        # minimum allowable distance
                        dist_min = entity_a.size + entity_b.shape[1] + dumping
                        # softmax penetration
                        k = self.contact_margin
                        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
                        force = 2*(self.contact_force *
                                   delta_pos / dist * penetration)
                        force_a = np.array(
                            [0, force[1]]) if entity_a.movable else None
                        force_b = np.zeros(2)
                        return [force_a, force_b]

                    # * if vertical collision
                    if(lowerline <= entity_a.state.p_pos[1] and
                            upperline >= entity_a.state.p_pos[1]):

                        # * apply vertical force
                        # minimum allowable distance
                        dist_min = entity_a.size + entity_b.shape[0] + dumping
                        # softmax penetration
                        k = self.contact_margin
                        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
                        force = 2*(self.contact_force *
                                   delta_pos / dist * penetration)
                        force_a = np.array(
                            [force[0], 0]) if entity_a.movable else None
                        force_b = np.zeros(2)
                        return [force_a, force_b]
                    pass
        # * if entity_a is a Agent & entity_b is a Agent
        if (isinstance(entity_a, Agent) and isinstance(entity_b, Agent)):

            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
            # * collison
            if(dist < dist_min):
                # print('Collision with agent')
                entity_a.isCollided = True
                entity_b.isCollided = True
                    
                entity_a.isWreck = True
                entity_b.isWreck = True


                pass
            # softmax penetration
            k = self.contact_margin
            penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
            force = self.contact_force * delta_pos / dist * penetration
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
            return [force_a, force_b]
        return [np.zeros(2), np.zeros(2)]
