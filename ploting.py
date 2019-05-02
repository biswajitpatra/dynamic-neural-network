import matplotlib.pyplot as plt
import numpy as np
import neuron as n
import math

error=1
thres=0.8
neus=[]
conns=[]
'''
plt.ion()
for i in range(50):
    y = np.random.random([10,1])
    plt.plot(y)
    plt.draw()
    plt.pause(0.00000000000000000000000001)
   # plt.clf()


'''
        
plt.ion()

def addneu(neu):
    global neus
    neus+=[neu]
def showneu(neu):
    plt.plot([neu.loc[0]],[neu.loc[1]],"ro")
def showcon(co):
    weight=co[0].inneu[co[1]]
    lw=  abs(math.tanh(weight))*4
    if weight>0:
        color='red'
    else:
        color='blue'
    plt.plot([co[0].loc[0],co[1].loc[0]],[co[0].loc[1],co[1].loc[1]],linewidth=lw,color=color)
    
def show():
    while True:
      for conn in conns:
          showcon(conn)
      for ne in neus:
          showneu(ne)
      plt.bar(-1,error*50,bottom=2,width=0.2,edgecolor="k")    
      plt.draw()
      plt.pause(1)
      plt.clf()
def addline(neu1,neu2):
    global conns
    conns+=[[neu1,neu2]]
