#agentを表すsuperclass
from abc import ABCMeta,abstractmethod

class Algorithm:
    def __init__(self,state_shape,act_num):

        self.state_shape=state_shape # 入力次元数
        self.act_num=act_num # 出力次元数
        
    # ニューラルネットワークを作成する関数
    @abstractmethod
    def build_network(self):
        pass
        
    # ニューラルネットワークの出力
    @abstractmethod
    def network_output(self):
        pass

    # 学習を行う関数
    @abstractmethod
    def update(self):
        pass
