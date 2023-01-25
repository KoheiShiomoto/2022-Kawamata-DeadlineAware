import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import os
import shutil
import time

import agents
from jobsets import jobset

def main():
    #引数設定
    parser=argparse.ArgumentParser(description="This program shows a scheduling simulation")
    parser.add_argument("--make",action="store_true",help="make a Job Set")
    parser.add_argument("--train",action="store_true",help="Run training")
    parser.add_argument("--test",action="store_true",help="Run test")
    parser.add_argument("--seed",default=1,type=int,help="The seed value for the random number. The default value is 1.")
    parser.add_argument("--capacity","-c",default=10.0,type=float,help="Line speed [Gbps]. The default is 10[Gb/s].")
    parser.add_argument("--delta","-d",default=0.1,type=float,help="Unit slot length (scheduling cycle length)[sec] The default is 0.1[sec]")
    parser.add_argument("--job_size",default=10.0,type=float,help="Average job processing time[Gbits] The default is 10[Gbits]")
    parser.add_argument("--step","-s",default=200,type=int,help="Job set episode length")
    parser.add_argument("--link","-l",default=0.9,type=float,help="Load The default is 0.9")
    parser.add_argument("--deadline_min","-i",default=1.1,type=float,help="Indicates the minimum value of deadline, which is several times the job size")
    parser.add_argument("--deadline_max","-x",default=5.0,type=float,help="Indicates the maximum value of deadline, which is several times the job size")
    parser.add_argument("--job_distribution",default="exponential",help="The probability distribution used to create the job set. exponential, uniform, pareto are supported.")
    parser.add_argument("--job_set_parameter",default=None,type=str,help="Job Set Path Specification")
    parser.add_argument("--model_parameter",default=None,type=str,help="File path specification for the model")
    parser.add_argument("--weight","-w",default=None,type=str,help="Used to specify the file path for model weights")
    parser.add_argument("--act_num","-a",default=5,type=int,help="Number of jobs considered by the scheduler")
    parser.add_argument("--episode",default=2000,type=int,help="Number of deep reinforcement learning")
    parser.add_argument("--iterate",default=200,type=int,help="Simulation iteration count")
    parser.add_argument("--init_alg",default="EDF",help="Algorithm used to learn the initial policy")
    parser.add_argument("--init_num",default=200,type=int,help="Number of episodes used for initial policy learning")
    parser.add_argument("--test_alg",default="proposal",help="Algorithm used in test")
    parser.add_argument("--bandwith_distribution",default=None,help="Probability distribution used for bandwidth control. Poisson and uniform are supported.")
    
    args=parser.parse_args()
    seed=args.seed
    last_arrival_step=args.step
    link_utilization=args.link
    capacity=args.capacity*1.0e+9 # [Gb/s]を[b/s]に変換
    delta=args.delta
    job_size=args.job_size*1.0e+9 # [Gbits]を[bits]に変換
    deadline_min=args.deadline_min
    deadline_max=args.deadline_max
    job_distribution=args.job_distribution
    job_set_parameter=args.job_set_parameter
    model_parameter=args.model_parameter
    weight=args.weight
    act_num=args.act_num
    episode=args.episode
    iterate=args.iterate
    init_alg=args.init_alg
    init_num=args.init_num
    test_alg= args.test_alg
    bandwith_distribution=args.bandwith_distribution

    # 乱数の初期化
    rd.seed(seed)
    print("initialize numpy seed to {}".format(seed))

    # 時間計測
    start_time=time.time()

    # ジョブセット作成
    if args.make:
        file_parameter_name=str(episode)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_d"+str(deadline_min)+"-"+str(deadline_max)
        job_class=jobset.JobClass(episode,capacity,delta,job_size,last_arrival_step,link_utilization,deadline_min,deadline_max,seed,job_distribution)
        job_class.make_job_set() # ジョブセット作成
        file_path="datasets/proposal/"+file_parameter_name+"/"

    # 学習
    elif args.train:
        if job_set_parameter==None:
            job_set_parameter=str(episode)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_d"+str(deadline_min)+"-"+str(deadline_max)
        file_path="datasets/proposal/"+job_set_parameter+"/"
        file_parameter_name=str(episode)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_d"+str(deadline_min)+"-"+str(deadline_max)+"_"+str(job_distribution)
        
        agent=agents.train.TrainAgent("policy gradient",init_alg,init_num,episode,last_arrival_step,link_utilization,act_num,seed,bandwith_distribution,file_path,file_parameter_name)
        agent.training()

        end_time=time.time()-start_time
        f=open("training_time"+init_alg+".txt","a")
        f.write(str(end_time))
        f.write("\n")
        f.write(str(file_path))
        f.write(str(file_parameter_name))
        f.write("\n")
        f.close()

        file_path="models/"+job_set_parameter+"/"
        os.makedirs(file_path,exist_ok=True)
        for i in glob.glob("*.hdf5"):
            shutil.move(i,file_path)

        if init_alg=="None":
            file_parameter_name="model"+str(episode)+"_Nact"+str(act_num)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_init"+init_alg+"0"
        else:
            file_parameter_name="model"+str(episode)+"_Nact"+str(act_num)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_init"+init_alg+str(init_num)
        file_path="logs/"+file_parameter_name+"/train/" 

    # テスト
    elif args.test:
        if job_set_parameter==None:
            job_set_parameter=str(iterate)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_d"+str(deadline_min)+"-"+str(deadline_max)
        file_path="datasets/proposal/"+job_set_parameter+"/"
        file_parameter_name=str(iterate)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_d"+str(deadline_min)+"-"+str(deadline_max)+"_"+str(job_distribution)
        if test_alg=="proposal":
            if model_parameter==None:
                nn_weight="models/"+weight
            else:
                nn_weight="models/"+model_parameter+"/"+weight
        else:
            nn_weight=weight
        agent=agents.test.TestAgent("policy gradient",nn_weight,test_alg,iterate,last_arrival_step,link_utilization,act_num,seed,bandwith_distribution,file_path,file_parameter_name)
        agent.test()

        end_time=time.time()-start_time
        f=open("time"+job_set_parameter+".txt","a")
        f.write(str(end_time))
        f.write("\n")
        f.write(str(file_path))
        f.write(str(file_parameter_name))
        f.write("\n")
        f.close()
        if test_alg=="proposal":
            file_parameter_name=weight
        else:
            file_parameter_name=job_set_parameter+test_alg
        file_path="logs/"+file_parameter_name+"/test/" 
    
    print("simulator is success.")
    os.makedirs(file_path,exist_ok=True)
    for i in ["*.pickle","*.png","*.csv","*.txt"]:
        for j in glob.glob(i):
            shutil.move(j, file_path)

if __name__=="__main__":
        main()
