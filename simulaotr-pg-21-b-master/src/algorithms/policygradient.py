import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.append("../")
import tensorflow as tf

from superclasses import algorithm

class PolicyGradient(algorithm.Algorithm):

    def __init__(self,state_shape,act_num):
        super().__init__(state_shape,act_num)
        # 環境の状態の形式
        self.state_shape=state_shape #入力次元数
        # 環境の取りうるaction数
        self.act_num=act_num+1 # 出力次元数(+1はパケットを転送しないという行動を取るため)
        self._act_num=act_num # 出力次元数

        self.learning_rate=1e-5 # 学習率
        self.discount_rate=0.9 # 割引累積報酬の計算に使用する係数γ
        self.layer_unit=256 # 隠れ層のユニット数
        self.seed=1
        
        self.epsilon=0.2 # 変動する確率の総和

        self.loss_log=[]

        #GPUメモリの制限
        physical_devices=tf.config.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device,True)
                tf.config.experimental.set_virtual_device_configuration(device,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*3)])
                print("{} memory growth: {}".format(device,tf.config.experimental.get_memory_growth(device)))
        else:
            print("Not enough GPU hardware devices available")
        
        # 重みの初期値を固定
        tf.random.set_seed(self.seed)
        print("initialize tensorflow seed to {}".format(self.seed))

    # ニューラルネットワークの作成
    def build_network(self):
        tf.keras.backend.clear_session()
        
        # ニューラルネットワークの作成，出力層をsoftmaxにすることで各行動の確率を出力
        input_=tf.keras.layers.Input(shape=self.state_shape)
        c=tf.keras.layers.Dense(self.layer_unit,activation="relu")(input_)
        c=tf.keras.layers.Dense(self.layer_unit,activation="relu")(c)
        c=tf.keras.layers.Dense(self.layer_unit,activation="relu")(c)
        c=tf.keras.layers.Dense(self.act_num,activation="softmax")(c) # 出力層がsoftmaxであることに注意(DQNだと異なる)
        self.model=tf.keras.models.Model(inputs=input_,outputs=c)
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate) # オプティマイザはAdamを使用
        # モデルのコンパイル
        self.model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

    def update(self,state_history,action_history,reward_history):
        one_hot_actions=tf.one_hot(action_history,self.act_num) # actionはone_hotベクトルにする
        with tf.GradientTape() as tape: # GradientTapeで勾配を計算
            action_probs=self.model(state_history,training=True) # 現在の戦略を獲得
            selected_action_probs=tf.reduce_sum(one_hot_actions*action_probs,axis=1) # 選択されたactionの確率を獲得，πθ(a|s)πθ(a|s) を計算
            clipped_probs=tf.clip_by_value(selected_action_probs,1e-10,1.0) # log(0)を回避
            loss=tf.reduce_mean(-tf.math.log(clipped_probs)*reward_history) # 期待値 logπθ(a|s)Qπθ(s,a)logπθ(a|s)Qπθ(s,a)を計算，最大化するために符号をマイナスにする
        # 勾配を元にoptimizerでモデルを更新
        gradients=tape.gradient(loss,self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))

        self.loss_log.append(loss)

    # ニューラルネットワークによる推論
    @tf.function
    def network_output(self,inputs,training):
        output=self.model(inputs,training=training)[0]
        return output

    def rl_select_action(self,inputs,wait_num,training):
        if wait_num<1:
            action=0
        else:
            output=self.network_output(inputs,training) # Nact+1次元
            act_prob=output.numpy()
            if wait_num<self._act_num: # 待っているジョブがNactより小さい
                for i in range(self._act_num-wait_num):
                    act_prob[wait_num+i]=0 # 確率を0にする
            act_prob=np.nan_to_num(act_prob) # Nanは0に置換
            if training:
                select_prob=act_prob/np.sum(act_prob)
                action=np.random.choice([i for i in range(self.act_num)],p=select_prob)
        
            else:
                action=np.argmax(act_prob)

        return action

    def explore_rl_select_action(self,inputs,wait_num,training):
        if wait_num<1:
            action=0
        else:
            output=self.network_output(inputs,training) # Nact+1次元
            act_prob=output.numpy()
            if wait_num<self._act_num: # 待っているジョブがNactより小さい
                for i in range(self._act_num-wait_num):
                    act_prob[wait_num+i]=0 # 確率を0にする
            act_prob=np.nan_to_num(act_prob) # Nanは0に置換
            if training:
                select_prob=act_prob/np.sum(act_prob)

                # self.epsilon:全体の保証できる確率の総和 epsilon*(wait_num+1) 
                epsilon=self.epsilon/(wait_num+1)
                select_prob=select_prob*(1-self.epsilon)

                for i in range(len(select_prob)):
                    if select_prob[i]!=0:
                        select_prob[i]=select_prob[i]+epsilon

                select_prob=select_prob/np.sum(select_prob)
                action=np.random.choice([i for i in range(self.act_num)],p=select_prob)
                
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
        one_hot_actions=tf.one_hot(action_history,self.act_num) # action_batchを one-hot表現にする
        self.model.fit(x=state_history,y=one_hot_actions,epochs=10,batch_size=1024,verbose=1)

    def loaded_model(self,nn_weight):
        self.build_network()
        self.model.load_weights(nn_weight+".hdf5") 
