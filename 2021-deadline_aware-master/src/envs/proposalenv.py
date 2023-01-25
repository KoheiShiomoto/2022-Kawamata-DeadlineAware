from collections import deque
import numpy as np
import numpy.random as rd
import pickle
import sys
sys.path.append("../")

from superclasses import environment

class ProposalEnv(environment.Env):
    def __init__(self,link_utilization,act_num,bandwith_distribution,file_path,file_parameter_name):
        super().__init__(act_num)
        self.act_num=act_num # バッファに溜まっているパケットの内何個まで考慮するのかの値
        self.state_shape=2*self.act_num # 残り転送量と残りバイト数の2つなのでact_numの2倍
        self.link_utilization=link_utilization

        self.reduction_rate=1.0 # 報酬の計算に用いる減衰のパラメータg

        self.bandwith_distribution=bandwith_distribution
        self.decrease_job=1.0 # 1タイムスロット時間で減らすジョブの量
        self.lamda_high=0.1
        self.total_bandwith=100 # 1/100スケールで乱数を振る

        self.busy_period_log=[]
        self.idol_period_log=[]
        self.arrival_in_busy_period_log=[]
        
        f=open(file_path+"job_set"+file_parameter_name+".pickle","rb")
        self.all_job_set=pickle.load(f)
        f.close()
        f=open(file_path+"arrivals"+file_parameter_name+".pickle","rb")
        self.all_arrivals=pickle.load(f)
        f.close()

    def set_job_set(self,i):
        self.job_set=self.all_job_set[i]
        self.arrivals=self.all_arrivals[i]

    # 環境の初期化
    def reset(self):
        
        self.current_step=-1 # observeメソッドで時間が1進むため，初期値は-1
        self.job_id=0 # ジョブのID
        self.job_log=[] # 到着したジョブの情報を保存

        self.buffer=[] # 到着したジョブを保存するバッファ

        self.busy_period=0
        self.idol_period=0
        self.arrival_in_busy_period=0

        # RRに関する変数
        self.next_act_num=0
        self.timeslot=1
        self.remain_timeslot=self.timeslot
        self.act_log=[]
        self.len_log=[]

    def observe(self):
        self.current_step+=1

        # ジョブの到着
        if self.current_step<len(self.arrivals):
            for _ in range(self.arrivals[self.current_step]):
                self.buffer.append([self.job_set[self.job_id][0],self.current_step+self.job_set[self.job_id][1],self.job_id])
                self.job_log.append([self.current_step+self.job_set[self.job_id][1],0])
                self.job_id+=1
                
        state=np.zeros(self.state_shape)
        self.wait_num=min(self.act_num,len(self.buffer))

        # ジョブの情報を保存
        for i in range(self.wait_num):
            state[i]=self.buffer[i][0]
            state[i+self.act_num]=self.buffer[i][1]
                
        return state

    # actionを実行して，結果を返す
    def step(self,action):
        done=False # ジョブを吐き出し終えたか
        completed_job_id=None # 終了したジョブのID

        #ジョブの処理
        if len(self.buffer)!=0:
            if self.bandwith_distribution==None:
                self.buffer[action][0]-=self.decrease_job
            elif self.bandwith_distribution=="poisson":
                high_bandwith=rd.poisson(self.lamda_high*self.total_bandwith,size=None)/self.total_bandwith
                if high_bandwith>1.0:
                    high_bandwith=1.0
                self.buffer[action][0]-=self.decrease_job*(1.0-high_bandwith)
            elif self.bandwith_distribution=="uniform":
                high_bandwith=rd.uniform(0,self.lamda_high*2)
                self.buffer[action][0]-=self.decrease_job*(1.0-high_bandwith)
            else:
                assert False,"invalid argument"

            # ジョブが終了したか
            if self.buffer[action][0]<=0:
                completed_job_id=self.buffer[action][2]
                self.job_log[completed_job_id][-1]=self.current_step
                del self.buffer[action]
                
            # busy period，idol periodの計測
            self.busy_period+=1
            if self.idol_period>0:
                self.idol_period_log.append(self.idol_period)
                self.idol_period=0
        else:
            self.idol_period+=1
            if self.busy_period>0:
                self.busy_period_log.append(self.busy_period)
                self.busy_period=0
                self.arrival_in_busy_period_log.append(self.arrival_in_busy_period)
                self.arrival_in_busy_period=0

        # 完了したら報酬を得る
        value=self.get_value(completed_job_id)

        # 終了判定
        if (self.current_step>len(self.arrivals))and(len(self.buffer)==0):
            done=True
        
        next_state=self.observe() # 次の状態に移行

        return value,next_state,done

    def get_value(self,completed_job_id):
        # ジョブが終了していないとき報酬は0
        if completed_job_id is None:
            value=0
        else:
            # 遅延の計算
            delay=min(0,self.job_log[completed_job_id][0]-self.job_log[completed_job_id][-1]) # d<0は遅延
            value=np.exp(delay*self.reduction_rate) # d>0が遅延とする場合はv=e^(-dg)
        
        return value

    def select_action(self,state,algorithm_name):
        action=0
        if algorithm_name=="EDF":
            MIN=1000000
            tmp=state[self.act_num:] # デッドラインの情報
            for i in range(len(tmp)):
                if tmp[i]!=0:
                    if tmp[i]<MIN:
                        MIN=tmp[i]
            if MIN==1000000:
                action=0

            else:
                action=np.where(tmp==MIN)[0][0]
        
        elif algorithm_name=="FIFO":
            action=0

        elif algorithm_name=="RR":

            tmp=state[self.act_num:] # デッドラインの情報
            tmp_len=0
            for i in range(len(tmp)):
                if tmp[i]!=0:
                    tmp_len+=1
            
            if tmp_len==0:
                self.next_act_num=0
                return 0
            
            if self.next_act_num>=tmp_len:
                self.next_act_num=0
            
            action=self.next_act_num

            if tmp[action]==1:
                self.remain_timeslot=self.timeslot
            else:
                self.remain_timeslot-=1
                if self.remain_timeslot==0:
                    self.next_act_num=action+1
                    self.remain_timeslot=self.timeslot

        return action

    def get_job_info(self):
        job_num=len(self.job_set)
        total_job_size = 0
        for i in range(job_num):
            total_job_size+=self.job_set[i][0]
        return job_num,total_job_size
