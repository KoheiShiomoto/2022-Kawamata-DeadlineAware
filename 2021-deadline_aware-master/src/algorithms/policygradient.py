import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
sys.path.append("../")
import tensorflow as tf

from superclasses import algorithm

class PolicyGradient(algorithm.Algorithm):

    def __init__(self,state_shape,act_num):
        super().__init__(state_shape,act_num)
        self.state_shape=state_shape # 入力次元数
        self.act_num=act_num # 出力次元数

        self.learning_rate=1e-5 # 学習率
        self.discount_rate=0.9 # 割引累積報酬の計算に使用する係数γ
        self.layer_unit=256 # 隠れ層のユニット数
        self.seed=1 # ニューラルネットワークの重みの初期値を固定

        self.loss_log=[]

        # GPUメモリの制限
        physical_devices=tf.config.list_physical_devices("GPU")
        if len(physical_devices)>0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device,True)
                tf.config.experimental.set_virtual_device_configuration(device,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*2)])
                print("{} memory growth: {}".format(device,tf.config.experimental.get_memory_growth(device)))
            
    # ニューラルネットワークの作成
    def build_network(self):
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.seed) # 重みの初期値を固定
        print("initialize tensorflow seed to {}".format(self.seed))
        # ニューラルネットワークの作成，出力層をsoftmaxにすることで各行動の確率を出力
        input_=tf.keras.layers.Input(shape=self.state_shape)
        c=tf.keras.layers.Dense(self.layer_unit,activation="relu")(input_)
        c=tf.keras.layers.Dense(self.layer_unit,activation="relu")(c)
        c=tf.keras.layers.Dense(self.layer_unit,activation="relu")(c)
        c=tf.keras.layers.Dense(self.act_num,activation="softmax")(c) # 出力層がsoftmaxであることに注意(DQNだと出力層は異なる)
        self.model=tf.keras.models.Model(inputs=input_,outputs=c)
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate) # オプティマイザはAdamを使用
        # モデルのコンパイル
        self.model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy']) #Policy Gradientの損失関数は，符号を変えれば交差エントロピー誤差と同じ

    def update(self,state_history,action_history,reward_history):
        one_hot_actions=tf.one_hot(action_history,self.act_num) # actionをone_hotベクトルにする
        # GradientTapeで勾配を計算
        with tf.GradientTape() as tape: 
            action_probs=self.model(state_history,training=True) # 現在の戦略を獲得
            selected_action_probs=tf.reduce_sum(one_hot_actions*action_probs,axis=1) # 選択されたactionの確率を獲得，πθ(a|s)πθ(a|s) を計算
            clipped_probs=tf.clip_by_value(selected_action_probs,1e-10,1.0) # log(0)を回避
            loss=tf.reduce_mean(-tf.math.log(clipped_probs)*reward_history) # 期待値 logπθ(a|s)Qπθ(s,a)logπθ(a|s)Qπθ(s,a)を計算，最大化するために符号をマイナスにする
        # 勾配を元にoptimizerでモデルを更新
        gradients=tape.gradient(loss,self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))

        self.loss_log.append(loss)

    #ニューラルネットワークによる推論
    @tf.function
    def network_output(self,inputs,training):
        output=self.model(inputs,training=training)[0] # 推論を行い，状態遷移確率を要素に持つTensorを返す
        return output

    def rl_select_action(self,inputs,wait_num,training):
        if wait_num<2:
            action=0
        else:
            output=self.network_output(inputs,training)
            act_prob=output[:wait_num]
            act_prob=act_prob.numpy()
            if training:
                select_prob=act_prob/np.sum(act_prob)
                action=np.random.choice([i for i in range(wait_num)],p=select_prob)
        
            else:
                action=np.argmax(act_prob)

        return action

    # 割引累積報酬を返す
    def calculate_discount_rewards(self,rewards):
        rewards_len=len(rewards)
        discount_rewards=[0 for i in range(rewards_len)]
        for t in range(rewards_len):
            discount_rewards[t]=sum([j*(self.discount_rate**i) for i,j in enumerate(rewards[t:])])

        return discount_rewards

    def plot_loss(self,file_parameter_name):
        f=open("./loss"+file_parameter_name+".pickle","wb")
        pickle.dump(self.loss_log,f)
        f.close()
        
        x=[i for i in range(len(self.loss_log))]
        plt.plot(x,self.loss_log)
        plt.xlabel("episode")
        plt.ylabel("loss")
        plt.savefig("./loss"+file_parameter_name+".png") 
        plt.close()

    def delate_memory(self):
        tf.keras.backend.clear_session()

    def initial_training(self,states,actions):
        state_history=np.array(states)
        action_history=np.array(actions)
        one_hot_actions=tf.one_hot(action_history,self.act_num) # aaction_historyを one-hot表現にする
        self.model.fit(x=state_history,y=one_hot_actions,epochs=10,batch_size=1024,verbose=1)

    def loaded_model(self,nn_weight):
        self.model.load_weights(nn_weight+".hdf5") # load_modelだとGPUメモリバカ食いするので重みだけ保存
        
