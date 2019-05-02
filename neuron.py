import ploting as p
import random
from datetime import datetime
random.seed(datetime.now())
learn_rate=0.1
threshold_his=0.005
max_waiting=30

def disconnectneu(n1,n2):
    if n1 in n2.outneu:
        p.conns.remove([n1,n2])
        n2.outneu.remove(n1)
        del n1.inneu[n2]
    else:
        p.conns.remove([n2,n1])
        n1.outneu.remove(n2)
        del n2.inneu[n1]

def fullcon(l1,l2):
    for l in l2:
        if l.bias!=None:
            continue
        for x in l1:
            l.addinput(x)
    
def full_conn(*args):
    for x in range(len(args)-1,0,-1):
        fullcon(args[x-1],args[x])
def set_all_func(func,deffunc):
    for x in p.neus:
        x.fun=func
        x.dfun=deffunc
def get_output(args,inputneu):
    for n in p.neus:
        n.outcre=False
    for a in args:
        a.generatevalue(inputneu)
    for n in p.neus:
        n.outcre=False
    return [a.outputval for a in args]
def backpropagate(outputneuron,error):
   for n in p.neus:
       n.delta=0
   for x in range(len(outputneuron)):
       outputneuron[x].backprop(error[x])
   for i in p.neus:
       if i.inpu==False and i.bias==None:
           i.changeweight()
def train(outputneuron,inputdata,outputdata):
    error=0
    for x in range(len(inputdata)):
        #get_output(outputneuron,inputdata[x])
        error+=sum([outputdata[x][l]-get_output(outputneuron,inputdata[x])[l] for l in range(len(outputdata[x]))])/len(outputdata[x])
        backpropagate(outputneuron,[outputdata[x][l]-get_output(outputneuron,inputdata[x])[l] for l in range(len(outputdata[x]))])
    p.error=error/len(inputdata)  
   
       
   
class neuron:
    def __init__(self,loc=None,fun=None,inpu=False,inpuno=-1,bias=None,deffunc=None):
        self.fun=fun
        self.dfun=deffunc
        self.inneu=dict()
        self.outneu=[]
        self.outputval=0
        self.outcre=False
        self.inpuno=inpuno
        self.inpu=inpu
        self.bias=bias
        self.delta=0
        self.hisweight=dict()  #for removing
        if loc!=None:
          self.loc=loc
          p.addneu(self)
    def addfunc(self,fun):
        self.fun=fun
    def addloc(self,loc):
        self.loc=loc
        p.addneu(self)
    def addinput(self,neu,wei=None):
        if wei==None:
            wei=random.randint(-500,500)/1000
        self.inneu.update({neu:wei})
        self.hisweight.update({neu:[wei]})   #for removing
        neu.addoutput(self)
        p.addline(self,neu)
    def addoutput(self,neu):
        self.outneu+=[neu]
    def changeweight(self):
        for i in self.inneu:
            self.inneu[i]=self.inneu[i]+learn_rate*self.delta*self.dfun(self.outputval)*i.outputval
            if len(self.hisweight[i])==max_waiting:                       #for removing
                  self.hisweight[i].pop()
                  self.hisweight[i].insert(0,self.inneu[i])
                  if all(abs(x)<threshold_his for x in self.hisweight[i]):
                      disconnectneu(self,i)
                      print("---------------------------disconnected",self,i)
                      break
            else:
                  self.hisweight[i]+=[self.inneu[i]]            #for removing
                  
    def backprop(self,err=None):
        if err!=None:
            self.delta+=err
        for x,y in self.inneu.items():
            x.backprop(self.delta*y)
    def generatevalue(self,inpus=None):
      if self.bias!=None:
          self.outputval=self.bias
          self.outcre=True
          return
      self.outputval=0
      if self.inpu==False:  
        for i,j in self.inneu.items():
          if i.outcre==False:
              i.generatevalue(inpus)
          self.outputval+=j*i.outputval
        self.outputval=self.fun(self.outputval)
      elif self.inpu==True:
          self.outputval=inpus[self.inpuno]
      elif self.bias==True:
          self.outputval=1
      self.outcre=True    
        
        
