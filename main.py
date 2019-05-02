import ploting as p
import neuron as n
import threading
from time import sleep
import math
threading.Thread(target=p.show).start()

def tanhdiff(i):
    return 1-pow(i,2)

n1=n.neuron([0,2],inpu=True,inpuno=1)
n11=n.neuron([0,3],inpu=True,inpuno=2)
n12=n.neuron([0,1],inpu=True,inpuno=0)
nl3=n.neuron([0,4],inpu=True,inpuno=3)
nbi1=n.neuron([0,5],bias=1)

n2=n.neuron([1,1])
n3=n.neuron([1,2])
n4=n.neuron([1,3])
n5=n.neuron([1,4])
nbi2=n.neuron([1,5],bias=1)


ns1=n.neuron([2,1])
ns2=n.neuron([2,2])
ns3=n.neuron([2,3])
ns4=n.neuron([2,4])
ns5=n.neuron([2,5])
nbi3=n.neuron([2,5],bias=1)

nb1=n.neuron([3,1])
nb2=n.neuron([3,2])
nb3=n.neuron([3,3])
nb4=n.neuron([3,4])

no=n.neuron([4,2])
n.full_conn([n1,n11,n12,nl3,nbi1],[n2,n3,n4,n5,nbi2],[ns1,ns2,ns3,ns4,ns5,nbi3],[nb1,nb2,nb3,nb4],[no])
#n.set_all_func(math.tanh,tanhdiff)
n.set_all_func(lambda x:x if x>=0 else 0.01*x,lambda x:1 if x>=0 else 0.01)
#n.set_all_func(lambda x:1/(1+math.exp(-x)),lambda x:x*(1-x))
for x in range(50000):
 print(x)
 n.train([no],[[1,1,0,0],[1,0,0,1],[0,1,1,0],[0,0,1,1],[1,1,1,1],[0,0,0,0],[0,1,0,1],[1,0,1,0],[1,1,0,1],[0,1,0,0],[1,0,0,0]],[[1],[1],[1],[1],[1],[0],[1],[1],[1],[0],[0]])
print(n.get_output([no],[1,1,1,0]))






