import copy
import pickle
import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler
import sys
sys.path.append("../")
#import tensorflow as tf
import matplotlib.pyplot as plt

from superclasses import agent
#import algorithms

class TrainAgent(agent.Agent):

    def __init__(self,algorithm_name,init_alg,init_num,episode,last_arrival_step,link_utilization,act_num,seed,bandwith_distribution,file_path,file_parameter_name):
        super().__init__("proposal",algorithm_name,link_utilization,act_num,seed,bandwith_distribution,file_path,file_parameter_name)

        self.init_alg=init_alg

        if self.init_alg=="None":
            self.init_num=0
        else:
            self.init_num=init_num
        self.episode=episode
        self.last_arrival_step=last_arrival_step
        self.link_utilization=link_utilization
        self.act_num=act_num
        self.seed=seed

        self.algorithm.build_network()

    def training(self):
        # 初期方策の学習
        states=[]
        actions=[]
        rewards=[]

        for i in range(self.init_num):
            # ジョブセット初期化
            self.env.set_job_set(i)
            self.env.reset()
            
            # 現在の状態と次の状態
            next_state=None
            state=self.env.observe()
            
            # 環境の方でジョブの処理が終わるまでループ
            while True:
                if next_state is not None:
                    state=copy.copy(next_state)

                action=self.env.select_action(state,self.init_alg)
                value,next_state,done=self.env.step(action)

                states.append(state)
                actions.append(action)
                
                if done:
                    break

        # ニューラルネットワークが学習できるように変形
        if self.init_num!=0:
            self.algorithm.initial_training(states,actions)
        
        # 重みの保存
        file_parameter_name="0_Nact"+str(self.act_num)+"_T"+str(self.last_arrival_step)+"_"+str(self.link_utilization)+"_s"+str(self.seed)+"_init"+self.init_alg+str(self.init_num)
        if self.init_num!=0:
            self.algorithm.model.save_weights("./model"+file_parameter_name+".hdf5")

        # 初期ポリシー学習の重みを使用
        if self.init_num!=0:
            self.algorithm.loaded_model("model"+file_parameter_name)

        # 必要な情報を保存するリスト
        success_rate_log=[]
        job_num_log=[]
        total_job_size_log=[]
        in_buffer_log=[]
        completed_job_num_log=[]
        reward_log=[]

        # ニューラルネットワークの学習
        for j in range(self.episode):
            #acc_action=[self.act_num,self.act_num,self.act_num,1,1,1,1,1,0,0,0]
            states=[]
            actions=[]
            rewards=[]
            buffer_log=[]
            
            # ジョブセット初期化
            self.env.set_job_set(j)
            self.env.reset()
            
            # job_numとtotal_job_sizeに情報を保存
            job_num,total_job_size=self.env.get_job_info()
            job_num_log.append(job_num)
            total_job_size_log.append(total_job_size)

            next_state=None
            # 現在の状態をstateに代入
            state=self.env.observe()
            
            # 学習の初期段階ではパラメータ更新を早めるために任意の地点ジョブの処理を中断する
            while True:
                if next_state is not None:
                    state=copy.copy(next_state)
                
                #buffer_logに情報を保存
                buffer_log.append(len(self.env.buffer))
                if self.env.act_now==False:
                    if j<=self.episode*0.2:
                        action=self.algorithm.explore_rl_select_action(np.array([state]),self.env.wait_num,True)
                    else:
                        action=self.algorithm.rl_select_action(np.array([state]),self.env.wait_num,True)
                value,next_state,done=self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(value)

                # 終了
                if done:
                    # successrateを算出
                    reward,completed_job_num=self.calculate_reward(job_num,rewards)
                    completed_job_num_log.append(completed_job_num)
                    if job_num!=0:
                        success_rate=completed_job_num/job_num
                    else:
                        success_rate=0
                    success_rate_log.append(success_rate)
                    reward_log.append(reward)
                    print("itr: {}/{} total_rewards: {}".format(j+1,self.episode,reward))
                    break
                    
            in_buffer_log.append(buffer_log)
            
            # データ形式を変形
            state_history=np.array(states)
            action_history=np.array(actions)
            discount_rewards=self.algorithm.calculate_discount_rewards(rewards)
            reward_history=np.array(discount_rewards)
            # パラメータ更新
            self.algorithm.update(state_history,action_history,reward_history)

        # 学習終了時の各種データ保存(hdf5,pickle,csv)
        file_parameter_name=str(self.episode)+"_Nact"+str(self.act_num)+"_T"+str(self.last_arrival_step)+"_"+str(self.link_utilization)+"_s"+str(self.seed)+"_init"+self.init_alg+str(self.init_num)
        if self.episode>0:
            self.algorithm.model.save_weights("./model"+file_parameter_name+".hdf5")
            ave_buffer_log=self.get_in_buffer_average_list(in_buffer_log)
            self.save_csv(self.get_in_list_average(success_rate_log),self.get_in_list_average(job_num_log),self.get_in_list_average(total_job_size_log),self.get_in_list_average(ave_buffer_log),file_parameter_name)
            self.algorithm.plot_loss(file_parameter_name)
            self.plot_success_rate(success_rate_log,completed_job_num_log,job_num_log,file_parameter_name)
            self.plot_total_job_size(total_job_size_log,file_parameter_name)
            self.plot_ave_buffer(ave_buffer_log,file_parameter_name)
            self.plot_value(reward_log,file_parameter_name)
        
        self.algorithm.delate_memory()
            
    def calculate_reward(self,job_num,rewards):
        completed_job_num=0
        for i in range(job_num):
            delay=min(0,self.env.job_log[i][0]-self.env.job_log[i][-1])
            if delay>=0:
                completed_job_num+=1
        reward=sum(rewards)
        
        return reward,completed_job_num
    
    def plot_success_rate(self,success_rate_log,completed_job_num_log,job_num_log,file_parameter_name):
        f=open("./train_completed_job_num"+file_parameter_name+".pickle","wb")
        pickle.dump(completed_job_num_log,f)
        f.close()
        f=open("./train_success_rate_log"+file_parameter_name+".pickle","wb")
        pickle.dump(success_rate_log,f)
        f.close()
        
        total_completed_job_num=0
        total_job_num=0
        accumulated_success_rate_log=[]
        for i in range(len(success_rate_log)):
            total_completed_job_num+=completed_job_num_log[i]
            total_job_num+=job_num_log[i]
            if total_job_num!=0:
                accumulated_success_rate=total_completed_job_num/total_job_num
            else:
                accumulated_success_rate=0
            accumulated_success_rate_log.append(accumulated_success_rate)
        f=open("./train_success_rate"+file_parameter_name+".pickle","wb")
        pickle.dump(accumulated_success_rate_log,f)
        f.close()
        
        x=[i for i in range(len(accumulated_success_rate_log))]
        plt.plot(x,accumulated_success_rate_log)
        plt.yticks(np.arange(0,1.1,0.1))
        plt.xlabel("episode")
        plt.ylabel("success rate")
        plt.savefig("./train_success_rate_accumulated"+file_parameter_name+".png") 
        plt.close()

    def plot_value(self,value_log,file_parameter_name):
        f=open("./train_value"+file_parameter_name+".pickle","wb")
        pickle.dump(value_log,f)
        f.close()
        
        x=[i for i in range(len(value_log))]
        plt.plot(x,value_log)
        plt.xlabel("episode")
        plt.ylabel("value")
        plt.savefig("./train_value"+file_parameter_name+".png") 
        plt.close()

    def plot_total_job_size(self,total_job_size_log,file_parameter_name):
        f=open("./train_total_job_size"+file_parameter_name+".pickle","wb")
        pickle.dump(total_job_size_log,f)
        f.close()
        
        x=[i for i in range(len(total_job_size_log))]
        plt.plot(x,total_job_size_log)
        plt.xlabel("episode")
        plt.ylabel("job size")
        plt.savefig("./train_total_job_size"+file_parameter_name+".png") 
        plt.close()

    def get_in_buffer_average_list(self,in_buffer_log):
        ave_buffer_log=[]
        for i in range(len(in_buffer_log)):
            ave_buffer=0
            for j in range(len(in_buffer_log[i])):
                ave_buffer+=in_buffer_log[i][j]
                ave_buffer/=len(in_buffer_log[i])
                ave_buffer_log.append(ave_buffer)
                        
        return ave_buffer_log

    def plot_ave_buffer(self,ave_buffer_log,file_parameter_name):
        f=open("./train_ave_in_buffer_size"+file_parameter_name+".pickle","wb")
        pickle.dump(ave_buffer_log,f)
        f.close()
        
        x=[i for i in range(len(ave_buffer_log))]
        plt.plot(x,ave_buffer_log)
        plt.xlabel("episode")
        plt.ylabel("ave in buffer size")
        plt.savefig("./train_ave_in_buffer_size"+file_parameter_name+".png") 
        plt.close()

    def get_in_list_average(self,log):
        ave=0
        if len(log)>0:
            for i in log:
                ave+=i
            ave/=len(log)

        return ave

    def save_csv(self,ave_success_rate,ave_job_num,ave_total_job_size,ave_in_buffer,file_parameter_name):
        data=pd.DataFrame([ave_success_rate,ave_job_num,ave_total_job_size,ave_in_buffer],index=["ave success rate","ave job num","ave total job size","ave in buffer"])
        data=data.T
        data.to_csv("saveDATA"+file_parameter_name+".csv", index=False)

