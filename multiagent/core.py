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

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    '''
    Addition to entity state it has a communication state
    '''
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

#! action of the agent
class Action(object):
    '''
    Physical and communication action
    '''
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

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
    def __init__(self, pos = 'hor', shape = (0.2, 0.8)):
        super(Landmark, self).__init__()
        self.pos = pos
        self.shape = shape
    

#! properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
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
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        a = [type(entity) for entity in self.entities]
        
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                #! velocity calculation
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    #! Collision is wrong for square objects, need to reformulate this 
    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        #get collision
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        #* if entity_a is a Agent & entity_b is a Landmark
        if (isinstance(entity_a, Agent) and isinstance(entity_b, Landmark)):
            # print(type(entity_a),type(entity_b) )
            
            if(entity_b.pos == 'ver'):
                dumping = 0.01
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                if(np.abs(delta_pos[0]) - dumping <= entity_b.shape[0]/2 + entity_a.size):
                    if(np.abs(delta_pos[1]) <= entity_b.shape[1]/2 + entity_a.size):
                    #* collison
                        print('Collision')
                        dist = np.sqrt(np.sum(np.square(delta_pos)))
                        
                        upperline = entity_b.state.p_pos[1] + entity_b.shape[1]/2
                        lowerline = entity_b.state.p_pos[1] - entity_b.shape[1]/2
                        rightline = entity_b.state.p_pos[0] + entity_b.shape[0]/2
                        leftline  = entity_b.state.p_pos[0] - entity_b.shape[0]/2
                    
                        #* if horizontal collision
                        if(leftline <= entity_a.state.p_pos[0] and \
                            rightline >= entity_a.state.p_pos[0]):
                            #*change horizontal velocity
                            entity_a.state.p_vel = entity_a.state.p_vel*[-100,1]

                        #* if vertical collision
                        if(lowerline <= entity_a.state.p_pos[1] and \
                            upperline >= entity_a.state.p_pos[1]):
                            #*change vertical velocity
                            entity_a.state.p_vel = entity_a.state.p_vel*[1,-100]

                        pass
                        
        return [np.zeros(2) ,np.zeros(2)]