import Memory, Agent, Environment, DQN_Model, Result 

def main():

    memory = Memory.Memory()  
    agent = Agent.Agent()
    environment = Environment.Environment()
    model = DQN_Model.Model()
    results = Result.Result()

    eval_network = model.create_neural_network()
    target_network  = eval_network

    for episode in range(model.no_episodes):

        agent.reset_accumulated_rewards()
        agent.reset_action_counter()
        environment.reset_termination_flag()

        state = agent.observe_environment(environment)
        
        while not environment.terminated:
            
            consideration = agent.consider_actions(state, environment, model.current_epsilon, eval_network)
            next_state, reward, termination_flag = agent.take_action(environment, consideration)
            agent.accumulated_rewards += reward
            agent.action_counter += 1
            memory.store_in_memory(consideration, reward, state, next_state, termination_flag)
            
            state = next_state           
            model.update(eval_network, target_network, memory)
            model.adjust_target_network(target_network, eval_network, agent.action_counter)

            environment.terminated = termination_flag
            

        results.store_result(episode+1, agent.accumulated_rewards, agent.action_counter, model.current_epsilon)

    results.plot_results(environment)
    results.save_result_csv(environment)
    model.save_trained_nn(eval_network, environment)

if __name__ == "__main__":
    main()
