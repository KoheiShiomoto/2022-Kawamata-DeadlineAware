from abc import ABCMeta,abstractmethod

#強化学習における環境を表す
class Env:
    def __init__(self,act_num):

        self.act_num=act_num
        self.state_shape=2*act_num

    def in_out_info(self):
        return self.state_shape,self.act_num

    #状態を与えて行動を返す
    @abstractmethod
    def rl_select_action(self):
        pass
        
    #状態を返す
    @abstractmethod
    def observe(self):
        pass

    #次の状態に進める
    @abstractmethod
    def step(self):
        pass

    #初期状態にする
    @abstractmethod
    def reset(self):
        pass

    #行動に対する報酬を返す
    @abstractmethod
    def get_reward(self):
        pass