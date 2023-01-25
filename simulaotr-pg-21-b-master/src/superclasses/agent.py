#agentを表すsuperclass
from abc import ABCMeta,abstractmethod
import sys
sys.path.append("../")

import algorithms
import envs

class Agent:
    def __init__(self,env_name,algorithm_name,link_utilization,act_num,seed,bandwith_distribution,file_path,file_parameter_name):

        self.env_name=env_name
        self.algorithm_name=algorithm_name
        self.link_utilization=link_utilization
        self.seed=seed
        self.bandwith_distribution=bandwith_distribution
        self.act_num=act_num

        #オブジェクトの実体化
        if env_name=="proposal":
            self.env=envs.proposalenv.ProposalEnv(self.link_utilization,self.act_num,self.bandwith_distribution,file_path,file_parameter_name)
        else:
            assert False,"invalid env name"
        
        self.state_shape,self.act_num=self.env.in_out_info()

        if algorithm_name=="policy gradient":
            self.algorithm=algorithms.policygradient.PolicyGradient(self.state_shape,self.act_num)
        else:
            assert False,"invalid algorithm name"

        # エージェントを実行したときにする動作の抽象メソッド
        @abstractmethod
        def training(self):
            pass
        
        @abstractmethod
        def test(self):
            pass
