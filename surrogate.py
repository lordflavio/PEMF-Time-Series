from PEMF.Models_Trainer.svr_trainer import svr_trainer,predict_svr
from PEMF.Models_Trainer.dace_trainer import dece_treiner,predict_deca
class Surrogate:
    def __init__(self,Surrogate,HP,Kernel):
        self.surrogate = Surrogate
        self.hp = HP
        self.Kernel = Kernel
    def fit(self, x,y):
        if(self.surrogate =="SVR"):
            self.model = svr_trainer(x,y,self.hp,self.Kernel)
            return self.model
        elif(self.surrogate == "deca"):
            self.model = dece_treiner(x,y,self.hp,self.Kernel)
            return self.model
    def predict(self,x):
        if (self.surrogate == "SVR"):
            return predict_svr(self.model,x)
        elif (self.surrogate == "deca"):
            return predict_deca(self.model,x)
