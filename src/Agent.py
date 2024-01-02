import random
import numpy as np
import tensorflow as tf

class Agent():
    
    def __init__(self):
        self.accumulated_rewards = 0
        self.action_counter = 0

    def reset_accumulated_rewards(self) -> int:
        self.accumulated_rewards = 0

    def reset_action_counter(self) -> int:
        self.action_counter = 0

    def observe_environment(self, environment):
        observation, info = environment.gym.reset()
        return observation
    
    def explore(self, current_epsilon) -> bool:
        if current_epsilon > random.random():
            return True
        else:
            return False       
    
    def exploit(self, eval_network, state) -> np.array:
        possible_outcomes = eval_network.predict(np.array([state]))
        return np.argmax(possible_outcomes)

    def consider_actions(self, state, environment, current_epsilon, eval_network) -> np.array:
        if self.explore(current_epsilon) == True:
            return np.random.choice(environment.action_space)
        else:
            return self.exploit(eval_network, state)

    def take_action(self, environment, consideration):
        environment_feedback = environment.gym.step(consideration)
        next_observation = environment_feedback[0]
        reward = environment_feedback[1]
        termination = max(environment_feedback[2], environment_feedback[3])
        return next_observation, reward, termination
    
if __name__ == "__main__":
    agent = Agent()
