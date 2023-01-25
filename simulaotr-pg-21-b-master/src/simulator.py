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
    parser.add_argument("--make",action="store_true",help="ジョブセットの作成")
    parser.add_argument("--busy",action="store_true",help="学習かテストのどちらのデータを作成するか．Trueなら学習データを作成")
    parser.add_argument("--train",action="store_true",help="学習の実行")
    parser.add_argument("--test",action="store_true",help="testの実行")

    parser.add_argument("--seed",default=1,type=int,help="乱数のシード値 デフォルト値は1")

    #Typical Job Set Parameter
    #parser.add_argument("arg1",help="この引数の説明（なくてもよい）")    # 必須の引数を追加
    #parser.add_argument("--arg3")    # オプション引数（指定しなくても良い引数）を追加
    #parser.add_argument("--arg4","-a")   # よく使う引数なら省略形があると使う時に便利
    #parser.add_argument("--pjName","-f",default="netsoft2021-papers",help="入力CSVファイルの指定 拡張子の.csvは含めない名前")
    parser.add_argument("--capacity","-c",default=10.0,type=float,help="回線速度 単位は [Gbps] デフォルトは10[Gb/s]")
    parser.add_argument("--delta","-d",default=0.1,type=float,help="単位スロット長(スケジューリング周期長) 単位は[sec] デフォルトは0.1[sec]")
    parser.add_argument("--job_size",default=10.0,type=float,help="jobの平均サイズ 単位は[Gbits] デフォルトは10[Gbits]")
    parser.add_argument("--step","-s",default=200,type=int,help="ジョブ・セットのエピソード長")
    parser.add_argument("--link","-l",default=0.9,type=float,help="負荷 デフォルトは0.9")
    parser.add_argument("--deadline_min","-i",default=1.1,type=float,help="deadlineの最小値をjobSizeの何倍かを示す")
    parser.add_argument("--deadline_max","-x",default=13.0,type=float,help="deadlineの最大値をjobSizeの何倍かを示す")
    #parser.add_argument("--job_file","-j",default=None)

    #Typical Neural Network Parameter 
    parser.add_argument("--weight","-w",default=None)
    parser.add_argument("--act_num","-a",default=None,type=int)
    parser.add_argument("--episode",default=2000,type=int,help="深層強化学習の学習回数")
    parser.add_argument("--iterate",default=2000,type=int,help="シミュレーションの繰り返し回数")
    parser.add_argument("--init_alg",default="EDF",help="初期方策の学習で使用するアルゴリズム")
    parser.add_argument("--init_num",default=200,type=int,help="初期方策の学習回数")
    parser.add_argument("--test_alg",default="proposal",help="testで使用するアルゴリズム")
    
    args=parser.parse_args()
    seed=args.seed
    last_arrival_step=args.step
    link_utilization=args.link
    capacity=args.capacity*1.0e+9  # [Gb/s]を[b/s]に変換
    delta=args.delta
    job_size=args.job_size*1.0e+9  # [Gbits]を[bits]に変換
    deadline_min=args.deadline_min
    deadline_max=args.deadline_max
    #job_file=args.job_file
    weight=args.weight
    act_num=args.act_num
    episode=args.episode
    iterate=args.iterate
    init_alg=args.init_alg
    init_num=args.init_num
    test_alg= args.test_alg

    #乱数の初期化
    rd.seed(seed)
    print("initialize numpy seed to {}".format(seed))

    #時間計測
    start_time=time.time()

    #ジョブセット作成オプション
    if args.make:
        print("make jobset")
        file_parameter_name=str(episode)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_d"+str(deadline_min)+"-"+str(deadline_max)
        job_class=jobset.JobClass(episode,capacity,delta,job_size,last_arrival_step,link_utilization,deadline_min,deadline_max,seed)
        if args.busy:
            job_class.make_train_set() #ジョブセット作成
            file_path="datasets/train/"+file_parameter_name+"/"

        else:
            job_class.make_test_set() #ジョブセット作成
            #file_parameter_name=str(episode)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_d"+str(deadline_min)+"-"+str(deadline_max)
            file_path="datasets/test/"+file_parameter_name+"/"

    #学習オプション
    elif args.train:
        print("This simulation follows Tensorflow 2.2")
        print("train")
        file_parameter_name=str(episode+init_num)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_d"+str(deadline_min)+"-"+str(deadline_max)
        file_path="datasets/train/"+file_parameter_name+"/"
        
        agent=agents.train.TrainAgent("policy gradient",init_alg,init_num,episode,last_arrival_step,link_utilization,act_num,seed,file_path,file_parameter_name)
        agent.training()

        end_time=time.time()-start_time
        f=open("time.txt","a")
        f.write(str(end_time))
        f.write("\n")
        f.close()

        file_parameter_name="model"+str(episode)+"_Nact"+str(act_num)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_init"+init_alg+str(init_num)
        file_path="logs/"+file_parameter_name+"/" 

    #テストオプション
    elif args.test:
        print("This simulation follows Tensorflow 2.2")
        print("test")
        file_parameter_name=str(iterate)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_d"+str(deadline_min)+"-"+str(deadline_max)
        file_path="datasets/test/"+file_parameter_name+"/"
        if test_alg=="proposal":
            nn_weight="logs/"+weight+"/"+weight
        else:
            nn_weight=weight
        agent=agents.test.TestAgent("policy gradient",nn_weight,test_alg,iterate,last_arrival_step,link_utilization,act_num,seed,file_path,file_parameter_name)
        agent.test()

        end_time=time.time()-start_time
        f=open("time.txt","a")
        f.write(str(end_time))
        f.write("\n")
        f.close()
        if test_alg=="proposal":
            file_parameter_name=nn_weight
        else:
            file_parameter_name=str(iterate)+"_T"+str(last_arrival_step)+"_"+str(link_utilization)+"_s"+str(seed)+"_d"+str(deadline_min)+"-"+str(deadline_max)+test_alg
        file_path="logs/"+file_parameter_name+"/" 
    
    print("simulator is success.")
    os.makedirs(file_path,exist_ok=True)
    for i in ["*.pickle","*.png","*.csv","*.txt","*.hdf5"]:
        for j in glob.glob(i):
            shutil.move(j, file_path)

if __name__=="__main__":
        main()