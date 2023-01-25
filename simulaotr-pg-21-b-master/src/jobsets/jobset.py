import copy
import math
import numpy as np
import numpy.random as rd
import pickle

class JobClass:

    def __init__(self,episode,capacity,delta,job_size,last_arrival_step,link_utilization,deadline_min,deadline_max,seed,job_distribution):
        self.test_num=episode# 評価ジョブセットシミュレーション数
        self.seed=seed
        self.capacity=capacity # 回線速度　単位は [bps] デフォルトは10Gb/s
        self.delta=delta # 単位スロット長(スケジューリング周期長)　単位は[sec] デフォルトは0.1sec
        self.last_arrival_step=last_arrival_step # Ttrain,Ttest
        self.link_utilization=link_utilization # 回線使用率(負荷率)ρ デフォルトは0.9
        self.job_size=job_size # jobの平均サイズ　単位は[bits] デフォルトは10Gbits
        self.interval=self.job_size/self.capacity/self.link_utilization#到着間隔
        self.deadline_min=deadline_min  # deadlineの最小値をjob_sizeの何倍かを示す．単位は無名数 デフォルトは1.1
        self.deadline_max=deadline_max  # deadlineの最大値をjob_sizeの何倍かを示す．単位は無名数 デフォルトは13
        self.job_distribution=job_distribution
        print("capacity is {} [Gb/s].".format(self.capacity*1e-9))
        print("delta is {} sec.".format(self.delta))
        print("average job size is {} [Gbits].".format(self.job_size*1e-9))
        print("link utilizaation is {}.".format(self.link_utilization))
        print("average interval is {} [sec].".format(self.interval))
        print("deadline is between {} and {} times of job size.".format(deadline_min,deadline_max))
        rd.seed(self.seed)

        self.scale=self.job_size/self.capacity
        self.range=self.scale*2-0.1

    # job_size, deadlineを単位スロット長で返す
    # jobの到着時刻が来たら呼び出す
    def long_job_size_deadline(self):
        job_size=16
        deadline=80
        self.job_set.append([job_size,deadline])
        return job_size,deadline

    def short_job_size_deadline(self):
        job_size=3
        deadline=4
        self.job_set.append([job_size,deadline])
        return job_size,deadline

    # jobの到着時間を単位スロット等で返す
    def get_arrival_time(self): 
        time=rd.exponential(self.interval,size=1)
        arrival_time=(math.ceil(time/self.delta))
        return arrival_time

    # ジョブセットのジョブ数とジョブの合計サイズを返す.
    def get_job_info(self):
        total_job_size=0
        for i in range(self.job_num):
            total_job_size += self.job_set[i][0]
        return self.job_num,total_job_size

    # ジョブセットを生成する
    def busy_period_generate(self):
        time=0 # 現在時刻
        backlog=0 # ジョブが今どれだけ溜まってるか
        job_num=0
        self.job_set=[]
        self.arrival_list=[]
        period=0
        
        while True:
            _flag=0
            while True:
                job_num+=1 # ジョブを作成
                arrival_time=time # ジョブの到着時刻

                # 短いジョブと長いジョブ
                if _flag==0:
                    job_size,deadline=self.long_job_size_deadline()
                    interval=3

                elif _flag==6:
                    job_size,deadline=self.long_job_size_deadline()
                    interval=3

                else:
                    job_size,deadline=self.short_job_size_deadline()
                    if _flag==5:
                        interval=21
                    else:
                        interval=3
                self.arrival_list.append(arrival_time)
                _flag+=1
                time+=interval # 次の時刻をintervalだけ増やす
                backlog+=(job_size-interval) # 今のジョブサイズからintervalだけ減ったらbacklog
                if backlog<=0:
                    period=time+backlog # backlogが0になる時刻
                    backlog=0
                    _flag=0
                    break

            if period>=self.last_arrival_step: # 指定したタイムスロット時間超えたか
                break
                
        self.arrivals=np.zeros(period,dtype=int)
        for m in range(job_num):
            t=self.arrival_list[m]
            self.arrivals[t]+=1 # ジョブが到着しているなら1加算，到着してないなら0
            
        self.job_num=job_num

        return self.job_set

    # ジョブセット作成
    def make_job_set(self):
        train_arrivals=[]
        train_job_set=[]
        for i in range(self.test_num):
            self.busy_period_generate()
            train_arrivals.append(self.arrivals)
            train_job_set.append(self.job_set)
            print("generate: {}/{}".format(i+1,self.test_num))
        
        file_parameter_name="sample"
        f=open("./job_set"+file_parameter_name+".pickle", "wb")
        pickle.dump(train_job_set,f)
        f.close()
        f=open("./arrivals"+file_parameter_name+".pickle", "wb")
        pickle.dump(train_arrivals,f)
        f.close()
