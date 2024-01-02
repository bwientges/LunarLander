import matplotlib.pyplot as plt
import datetime
from configparser import ConfigParser
import pandas as pd
from pathlib import Path

class Result():

    def get_config(self):
            config = ConfigParser()
            path = Path(Path.cwd() / "config" / "config.ini")
            config.read(path)
            return config

    def __init__(self):
        self.result = []
        self.result_config = self.get_config()
        self.min_rewards_for_solving = int(self.result_config["Environment"]["accumulated_rewards_for_solving"])
        self.env_name = str(self.result_config["Environment"]["environment_name"])
        self.model_name = str(self.result_config["Model"]["model_name"])
        self.episode_nos = int(self.result_config["Model"]["number_of_episodes"])
        self.optimizer = str(self.result_config[self.model_name]["optimizer"])
        self.loss_function = str(self.result_config[self.model_name]["loss_function"])
        self.eps_decay = str(self.result_config["Model"]["epsilon_decay"])
        self.results = []  
        self.moving_average_window = int(self.result_config["Result"]["moving_average_window"])
        self.result_frame = pd.DataFrame(columns=["Episode", "Accumulated Rewards", f"Average Result over {self.moving_average_window} Episodes", "Action Counter", "Epsilon Value"])

    def calculate_moving_average(self, episode, accumulated_rewards) -> float:
        self.results.append(accumulated_rewards)
        if episode <= self.moving_average_window:
            return float(sum(self.results[:episode]) / episode)
        else:
            return float(sum(self.results[(episode-self.moving_average_window) : episode]) / self.moving_average_window) 

    def store_result(self, episode, accumulated_rewards, action_counter, epsilon):
        moving_average = self.calculate_moving_average(episode, accumulated_rewards)
        self.result.append((episode, accumulated_rewards, moving_average, action_counter, epsilon))
        self.result_frame.loc[len(self.result_frame)] = [episode, accumulated_rewards, moving_average, action_counter, epsilon]
        print(f"Stored Result - Episode: {episode}, Accumulated Rewards: {int(accumulated_rewards)}, Average Rewards over {self.moving_average_window} Episodes: {int(moving_average)}, No. of Actions: {action_counter}")

    def save_result_csv(self, environment):
        time = str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
        path = Path(Path.cwd() / "data" / str("result_csv_" + str(self.model_name)+"_"+str(self.episode_nos) + "_eps_" +str(environment.randomization_status) + "_env_"  + time + ".csv")) 
        self.result_frame.to_csv(path, sep=",")

    def check_for_solving_points(self, result_tuple) -> bool:
         if result_tuple[1] >= 200:
              return True
         else:
              return False

    def plot_results(self, environment):
        all_episodes = []
        average_accumulated_rewards = []
        solved_episodes = [] 
        solved_accumulated_rewards = []
        unsolved_episodes = []
        unsolved_accumumlated_rewards = []
        for result_tuple in self.result:
            all_episodes.append(int(result_tuple[0]))
            average_accumulated_rewards.append(int(result_tuple[2]))
            if self.check_for_solving_points(result_tuple):
                solved_episodes.append(int(result_tuple[0]))
                solved_accumulated_rewards.append(int(result_tuple[1]))        
            else:
                unsolved_episodes.append(result_tuple[0])
                if result_tuple[1] > -200: 
                    unsolved_accumumlated_rewards.append(result_tuple[1])
                else:
                    unsolved_accumumlated_rewards.append(-200) #capping negative accumulated rewards for clearer visualization in plot

        plt.figure(figsize=(12,6))
        plt.axis([0, self.episode_nos, min(unsolved_accumumlated_rewards), max(solved_accumulated_rewards)+50])
        plt.xlabel("Episodes")
        plt.ylabel("(Average) Accumulated Rewards")
        suptitle = f"{self.env_name} utilizing {self.model_name} in a {environment.randomization_status} Environment"
        plt.suptitle(suptitle, fontsize="14")
        title = f"Episodes: {self.episode_nos}, Optimizer: {self.optimizer}, Loss Function: {self.loss_function}, Epsilon Decay: {self.eps_decay}"
        plt.title(title, loc ="left", fontsize="12")
        plt.plot([i for i in range(1,(self.episode_nos+1),1)], [200 for entry in range(1,(self.episode_nos+1),1)], linestyle="--", linewidth=1, color="black", label="Divider Solved / Unsolved" )
        plt.plot(all_episodes,average_accumulated_rewards, linestyle="-", color="black", label=f"Average of last {self.moving_average_window} Rewards per Episode")
        plt.plot(solved_episodes, solved_accumulated_rewards, linestyle="", marker="^", markersize=5, color="black", label="Solved Episodes")
        plt.plot(unsolved_episodes, unsolved_accumumlated_rewards,linestyle="", marker="s", markersize=4, color="black", label="Unsolved Episodes")
        time = str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
        plt.legend()
        plt.savefig(fname=Path(Path.cwd() / "data" / str("result_plot_" + str(self.model_name) + "_"+ str(self.episode_nos) + "_eps_" + str(environment.randomization_status) + "_env_" + time + ".svg")), format="svg")

if __name__ == "__main__":
    result = Result()