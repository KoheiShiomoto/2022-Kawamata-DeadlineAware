import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append("../")

from superclasses import agent

class TestAgent(agent.Agent):
    def __init__(self,algorithm_name,nn_weight,test_alg,iterate,last_arrival_step,link_utilization,act_num,seed,bandwith_distribution,file_path,file_parameter_name):
        super().__init__("proposal",algorithm_name,link_utilization,act_num,seed,bandwith_distribution,file_path,file_parameter_name)

        self.test_alg=test_alg
        self.num=iterate
        self.last_arrival_step=last_arrival_step
        self.link_utilization=link_utilization
        self.act_num=act_num
        self.seed=seed

        if test_alg=="proposal":
            self.algorithm.build_network()
            self.algorithm.loaded_model(nn_weight)

        self.reward_log=[]
        self.success_rate_log=[]

    def test(self):
        
        job_num_log=[]
        completed_job_num_log=[]

        for i in range(self.num):
            rewards=[]
            self.env.set_job_set(i)
            self.env.reset()

            job_num,total_job_size=self.env.get_job_info()
            job_num_log.append(job_num)

            # 現在の状態と次の状態
            next_state=None
            state=self.env.observe()

            while True:
                if next_state is not None:
                    state=copy.copy(next_state)

                if self.test_alg=="proposal":
                    action=self.algorithm.rl_select_action(np.array([state]),self.env.wait_num,False)
                else:
                    action=self.env.select_action(state,self.test_alg)
                
                value,next_state,done=self.env.step(action)
                rewards.append(value)

                if done:
                    # successrateを算出
                    reward,total_delay,completed_job_num=self.calculate_reward(rewards)
                    completed_job_num_log.append(completed_job_num)
                    if job_num!=0:
                        success_rate=completed_job_num/job_num
                    else:
                        success_rate=0
                    self.success_rate_log.append(success_rate)
                    self.reward_log.append(reward)
                    print("num: {} test: {}".format(self.num,i))
                    break
    
        if self.test_alg=="proposal":
            self.algorithm.delate_memory()
        
        busy_period_log=self.env.busy_period_log
        idol_period_log=self.env.idol_period_log
        arrival_in_busy_period_log=self.env.arrival_in_busy_period_log

        # busy periodの平均分散の計算
        ave_busy_period=self.get_in_list_average(busy_period_log)
        var_busy_period=self.get_in_list_variance(busy_period_log,ave_busy_period)
        # idol periodの平均分散の計算
        ave_idol_period=self.get_in_list_average(idol_period_log)
        var_idol_period=self.get_in_list_variance(idol_period_log,ave_idol_period)
        # busy period中に来た客数の平均分散
        ave_arrival_in_busy_period=self.get_in_list_average(arrival_in_busy_period_log)
        var_arrival_in_busy_period=self.get_in_list_variance(arrival_in_busy_period_log,ave_arrival_in_busy_period)

        file_parameter_name=str(self.num)+"_Nact"+str(self.act_num)+"_T"+str(self.last_arrival_step)+"_"+str(self.link_utilization)+"_s"+str(self.seed)+"_test"+self.test_alg
        self.plot_success_rate(self.success_rate_log,completed_job_num_log,job_num_log,file_parameter_name)
        self.plot_value(self.reward_log,file_parameter_name)
        
        data=pd.DataFrame([ave_busy_period,var_busy_period,ave_idol_period,var_idol_period,ave_arrival_in_busy_period,var_arrival_in_busy_period],index=["ave busy period","var busy period","ave idol period","var idol period","ave arrival in busy period","var arrival in busy period"])
        data=data.T
        data.to_csv("test_period_data"+file_parameter_name+".csv")

        # test result
        total_value=sum(self.reward_log)      
        data=pd.DataFrame([total_value,total_delay,self.accumulated_success_rate],index=["total rewards","total delay","accumulated success rate"])
        data=data.T
        data.to_csv("test_result_data"+file_parameter_name+".csv")

    def calculate_reward(self,rewards):
        completed_job_num=0
        total_delay=0
        for i in range(len(self.env.job_log)):
            delay=min(0,self.env.job_log[i][0]-self.env.job_log[i][-1])
            if delay>=0:
                completed_job_num+=1
            else:
                total_delay+=delay

        reward=sum(rewards)
        
        return reward,total_delay,completed_job_num
    
    def plot_success_rate(self,success_rate_log,completed_job_num_log,job_num_log,file_parameter_name):
        f=open("./test_completed_job_num"+file_parameter_name+".pickle","wb")
        pickle.dump(completed_job_num_log,f)
        f.close()
        f=open("./test_success_rate_log"+file_parameter_name+".pickle","wb")
        pickle.dump(success_rate_log,f)
        f.close()
        
        total_completed_job_num=0
        total_job_num=0
        self.accumulate_success_rate=0
        accumulated_success_rate_log=[]
        for i in range(len(success_rate_log)):
            total_completed_job_num+=completed_job_num_log[i]
            total_job_num+=job_num_log[i]
            if total_job_num!=0:
                self.accumulated_success_rate=total_completed_job_num/total_job_num
            else:
                self.accumulated_success_rate=0
            accumulated_success_rate_log.append(self.accumulated_success_rate)
        f=open("./test_success_rate"+file_parameter_name+".pickle","wb")
        pickle.dump(accumulated_success_rate_log,f)
        f.close()
        
        x=[i for i in range(len(accumulated_success_rate_log))]
        plt.plot(x,accumulated_success_rate_log)
        plt.yticks(np.arange(0,1.1,0.1))
        plt.xlabel("iterate")
        plt.ylabel("success rate")
        plt.savefig("./test_success_rate_accumulated"+file_parameter_name+".png") 
        plt.close()

    def plot_value(self,value_log,file_parameter_name):
        f=open("./test_value"+file_parameter_name+".pickle","wb")
        pickle.dump(value_log,f)
        f.close()
        
        x=[i for i in range(len(value_log))]
        plt.plot(x,value_log)
        plt.xlabel("episode")
        plt.ylabel("value")
        plt.savefig("./test_value"+file_parameter_name+".png") 
        plt.close()

    def get_in_list_average(self,log):
        ave=0
        if len(log)>0:
            for i in log:
                ave+=i
            ave/=len(log)

        return ave

    def get_in_list_variance(self,log,ave):
        var=0
        if len(log)>0:
            for i in log:
                var+=(i-ave)**2
            var/=len(log)

        return var
        
    def save_period_log(busy_period_log, idol_period_log, arrival_in_busy_period_log,file_parameter_name):
        f=open("./busy_period_log"+file_parameter_name+".pickle","wb")
        pickle.dump(busy_period_log,f)
        f=open("./idol_period_log"+file_parameter_name+".pickle","wb")
        pickle.dump(idol_period_log,f)
        f=open("./arrival_in_busy_period_log"+file_parameter_name+".pickle","wb")
        pickle.dump(arrival_in_busy_period_log,f)
