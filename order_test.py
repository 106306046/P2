import time
from datetime import datetime
from random import sample
from pathlib import Path



path = Path("./log.txt")

MI = ['Up','Down','Left','Right']
SSVEP = ['A','B','C','D','E']

def give_order(mi,ssvep):
    current_time = datetime.now()
    with open(path, 'a+') as f:
        f.write(str(current_time) + ' ' + mi + ' ' + ssvep + '\n')
        #time.sleep(1)

with open(path,'a+') as f:
    f.truncate(0)
    f.write('0 0 idle idle\n')





while 1 :
    print("input:")
    order = input()
    if order == 'A' or order == 'B' or order == 'C' or order == 'D' or order == 'E' :
        give_order('Down', order)
    elif order == '2':
        give_order('Down', 'idle')
    elif order == '4':
        give_order('Left', 'idle')
    elif order == '6':
        give_order('Right', 'idle')
    elif order == '8':
        give_order('Up', 'idle')
    elif order == '5':
        give_order('idle', 'idle')
    else:
        print('wrong order')


for i in range(3,0,-1):
    print(i)
    time.sleep(1)
print('start')

give_order('idle','D')

for i in range(5):
    give_order('Down','idle')

for i in range(7):
    give_order('idle','idle')

give_order('Down','E')

for i in range(10):
    give_order('Right','idle')

for i in range(7):
    give_order('idle', 'idle')

give_order('Down', 'E')

for i in range(10):
    give_order('Right','idle')

give_order('Down', 'A')
give_order('Down', 'A')
give_order('Down', 'A')