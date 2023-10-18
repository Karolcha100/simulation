import math as m
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


#funkcje pomocniczne

def esub5(x):
    return (2.72) ** (x - 5)

def log2(x):
    return m.log(x + 1, 2)

def booleanoutput(x):
    if(x<0):
        return False
    else:
        return True

def logisticalA(x,a):
    return a/((2.72) ** (x - a) + 1)

def direction(x):
    if(m.sin(x) < 0):
        if(m.cos(x)<0):
            return "w"
        else:
            return "d"
    else:
        if(m.cos(x)>=0):
            return "s"
        else:
            return "a"

def TFabsdiv(x):
    if (booleanoutput(x)):
        return 0.5
    else:
        return 1



class Cord:
    def __init__(self, x=0, y=0):
        self.xx = x
        self.yy = y

    def get_str(self):
        return "("+str(self.xx)+" ; "+str(self.yy)+")"

    def getx(self):
        return self.xx

    def gety(self):
        return self.yy

    def update(self,xu,yu):
        self.xx = xu
        self.yy = yu

class Constans:
    def __init__(self):
        self.MAX_Lo = 9
        self.MAX_shoot_duration = 9
        self.MAX_shoot_range = 9
        self.feromons = "ABCDEFGHIJKLMNOPQRSTVWXYZ"
        self.feromons_ph = {'A': 0.0, 'B': 0.04, 'C': 0.08, 'D': 0.12, 'E': 0.16, 'F': 0.2, 'G': 0.24, 'H': 0.28,'I': 0.32,
                           'J': 0.36, 'K': 0.4, 'L': 0.44, 'M': 0.48, 'N': 0.52, 'O': 0.56, 'P': 0.6, 'Q': 0.64, 'R': 0.68,
                           'S': 0.72, 'T': 0.76, 'V': 0.8, 'W': 0.84, 'X': 0.88, 'Y': 0.92, 'Z': 0.96}

class Evolutionary:
    def __init__(self):

        self.Max_Switched_Values = {"Sight": 2, "Moving": 3, "FeroEmission": 3, "EatingEffectivnes": 1} #FerroEmission = shot max range

        self.max_hp = 100

        self.maxSpeed = 3

        self.shot_max_range = 3

        self.capacity_energy = 100

        self.capacity_feromons = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0,
                                  'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0,
                                  'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0}

        self.eating_effectivnes = 1
        self.eating_range = 3

        #self.LoSight_Type = 1
        #self.LoSight = 1
        #self.RLoSight = self.LoSight_Type + self.LoSight

        self.mutations_amount = 1
        self.mutation_strenght = 1

        self.reproduction_divide = 0.5

class Fixed():
    def __init__(self, evolutionary, neural):

        self.switched_values_cost = {"Sight": 0, "Moving": 0, "FeroEmission": 0, "EatingEffectivnes": 0}

        self.const_cost_LoSight = evolutionary.LoSight + esub5(evolutionary.LoSight_Type)

        self.const_cost_maxSpeed = evolutionary.maxSpeed  # !!!
        self.const_cost_shot_range = esub5(evolutionary.shot_max_range)  # !!!
        self.const_cost_capacity = log2(evolutionary.capacity_energy + 10 * (evolutionary.capacityA + evolutionary.capacityB + evolutionary.capacityC + evolutionary.capacityD) + esub5(evolutionary.capacityK + 3))  # !!!
        self.const_cost_eating_effectivnes = esub5((evolutionary.eating_effectivnes / 5) + 3)

        self.const_cost_hp = evolutionary.max_hp

        self.Total_Lo_cost = neural.TFSi * self.const_cost_LoSight
        self.Total_other_cost = neural.TF1 * self.const_cost_maxSpeed + neural.TF2 * self.const_cost_shot_range + neural.TF3 * self.const_cost_capacity + neural.TF4 * self.const_cost_eating_effectivnes
        self.Total_const_cost = self.Total_Lo_cost + self.Total_other_cost + self.const_cost_hp


        self.FeroEmission_cost = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0,
                                  'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0,
                                  'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0}

        self.hp_production_cost_1_unit = self.Total_const_cost

        self.reproducing_cost = self.Total_const_cost

        self.num_of_children = 0

    def upadate_FeroEmi_costs(self, Emi_decisions, Emi_durations, Emi_range):
        costs = {}
        for fer in Emi_decisions:
            if Emi_decisions[fer]:
                costs[fer] = esub5(Emi_durations[fer] * Emi_range)
            else:
                costs[fer] = 0

        return costs

    def upadate_Switched_cost(self,max_values,switches):
        costs = {}

    def upadate_all(self,neural):
        self.FeroEmission_cost = self.upadate_FeroEmi_costs(neural.EmmisionDecision,neural.EmissionDuration,neural.shot_range)


    def add_children(self):
        self.num_of_children = self.num_of_children + 1

    def get_children(self):
        return self.num_of_children


class Neural:
    def __init__(self):

        self.Energy_Switch = {"Sight": 1,"Moving": 1,"FeroEmission": 1,"EatingEffectivnes": 1}

        self.EmmisionDecision = {'A': False, 'B': False, 'C': False, 'D': False, 'E': False, 'F': False, 'G': False,
                                 'H': False, 'I': False, 'J': False, 'K': False, 'L': False, 'M': False, 'N': False,
                                 'O': False, 'P': False, 'Q': False, 'R': False, 'S': False, 'T': False, 'V': False,
                                 'W': False, 'X': False, 'Y': False, 'Z': False}
        
        self.EmissionDuration = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0,
                                 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0,
                                 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0}

        self.shot_range = 0

        self.FeroProd_Decision = {'A': False, 'B': False, 'C': False, 'D': False, 'E': False, 'F': False, 'G': False,
                                  'H': False, 'I': False, 'J': False, 'K': False, 'L': False, 'M': False, 'N': False,
                                  'O': False, 'P': False, 'Q': False, 'R': False, 'S': False, 'T': False, 'V': False,
                                  'W': False, 'X': False, 'Y': False, 'Z': False}

        self.actual_speed = 0

        self.shot_direction = "w"  # w,a,s,d

        self.movement_direction = "w"  # w,a,s,d

        self.reproducing = False

class Storage:
    def __init__(self):
        self.hp = 100
        self.energy = 100
        self.feromons = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0,
                         'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0,
                         'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0}
        self.cell_ph = 0

    def check_ferocap(self,feromons_in_specie):
        for fer in feromons_in_specie:
            if feromons_in_specie[fer] > self.feromons[fer]:
                self.hp = self.hp - 1

    def summing_feromons_ph(self,feromons_ph, feromons_in_specie):
        ferosum = 0
        amountsum = 0

        for fer in feromons_in_specie:
            ferosum = ferosum + feromons_ph[fer] * feromons_in_specie[fer]
            amountsum = amountsum + feromons_in_specie[fer]
        if amountsum == 0:
            return 0
        else:
            return ferosum / amountsum

    def add_feromon(self,feromon,amount):
        self.feromons[feromon] = self.feromons[feromon] + amount
        return self.feromons[feromon]



class MojaSiec(nn.Module):
    def __init__(self):
        super(MojaSiec, self).__init__()
        self.warstwa_wejsciowa = nn.Linear(361+7, 8)  # Warstwa wejściowa: 400 wejść, 8 neuronów
        self.warstwa2 = nn.Linear(8, 8)  # Pierwsza warstwa ukryta: 8 wejść, 8 neuronów
        self.warstwa3 = nn.Linear(8, 8)  # Druga warstwa ukryta: 8 wejść, 8 neuronów
        self.warstwa_wyjsciowa = nn.Linear(8, 25)  # Warstwa wyjściowa: 8 wejść, 32 neurony wyjściowe

    def forward(self, x):
        x = torch.relu(self.warstwa_wejsciowa(x))  # Funkcja aktywacji ReLU w warstwie wejściowej
        x = torch.relu(self.warstwa2(x))  # Funkcja aktywacji ReLU w pierwszej warstwie ukrytej
        x = torch.relu(self.warstwa3(x))  # Funkcja aktywacji ReLU w drugiej warstwie ukrytej
        x = self.warstwa_wyjsciowa(x)
        return x

class Stats:
    def __init__(self,cord,evolutionary,storage,fixed):
        self.number = 0
        self.identity = ""
        self.observed = False

    def get_number(self):
        return self.number

    def get_identity(self):
        return self.identity

    def observe(self):
        self.observed = True

    def unobserve(self):
        self.observed = False

    def ifobserved(self):
        return self.observed










class Species:
    def __init__(self):
        self.cord = Cord()
        self.constans = Constans()
        self.evolutionary = Evolutionary()
        self.neural = Neural()
        self.storage = Storage()
        self.fixed = Fixed(self.evolutionary, self.neural)
        self.network = MojaSiec()
        self.stats = Stats(self.cord,self.evolutionary,self.storage,self.fixed)

    def doing(self, sun_map, L):
        area = Sight_detection_sun(sun_map,self.cord.xx,self.cord.yy,self.evolutionary.RLoSight,self.constans.MAX_Lo,L)

        area = area + [self.storage.hp,self.storage.energy,self.storage.A,self.storage.B,
                       self.storage.C,self.storage.D,self.storage.K]

        area_T = torch.tensor(area,dtype=torch.float32)

        orders = self.network(area_T)

        self.neural.TFSi = TFabsdiv(orders[0].item())
        self.neural.TF1 = TFabsdiv(orders[1].item())
        self.neural.TF2 = TFabsdiv(orders[2].item())
        self.neural.TF3 = TFabsdiv(orders[3].item())
        self.neural.TF4 = TFabsdiv(orders[4].item())

        self.neural.if_shootA = booleanoutput(orders[5].item())
        self.neural.if_shootA = booleanoutput(orders[6].item())
        self.neural.if_shootA = booleanoutput(orders[7].item())
        self.neural.if_shootA = booleanoutput(orders[8].item())
        self.neural.if_shootA = booleanoutput(orders[9].item())

        self.neural.shot_range = int(round(logisticalA(orders[10].item(),self.evolutionary.shot_max_range),0))

        self.neural.shot_durationA = int(round(logisticalA(orders[11].item(), self.evolutionary.shot_max_range),0))
        self.neural.shot_durationB = int(round(logisticalA(orders[12].item(), self.evolutionary.shot_max_range), 0))
        self.neural.shot_durationC = int(round(logisticalA(orders[13].item(), self.evolutionary.shot_max_range), 0))
        self.neural.shot_durationD = int(round(logisticalA(orders[14].item(), self.evolutionary.shot_max_range), 0))
        self.neural.shot_durationK = int(round(logisticalA(orders[15].item(), self.evolutionary.shot_max_range), 0))

        self.neural.productionA_decision = booleanoutput(orders[16].item())
        self.neural.productionA_decision = booleanoutput(orders[17].item())
        self.neural.productionA_decision = booleanoutput(orders[18].item())
        self.neural.productionA_decision = booleanoutput(orders[19].item())
        self.neural.productionA_decision = booleanoutput(orders[20].item())

        self.neural.actual_speed = int(round(logisticalA(orders[21].item(),self.evolutionary.maxSpeed),0))

        self.neural.shot_direction = direction(orders[22].item())

        self.neural.movement_direction = direction(orders[23].item())

        self.neural.reproducing = booleanoutput(orders[24].item())

        self.storage.energy = self.storage.energy - self.fixed.Total_const_cost

    def moving(self,L):
        self.cord.xx, self.cord.yy \
            = dircords(self.cord.xx,self.cord.yy,self.neural.movement_direction,self.neural.actual_speed,L)






class Observation:
    def __init__(self):
        self.informations = {}
        self.counting = {}

    def add_temp(self,id_code,x_loc,y_loc,color = "b",ticks = 1):
        self.informations[id_code] = [x_loc, y_loc, color]
        self.counting[id_code] = ticks

    def remove_temp(self,id_code):
        deleted_value1 = self.counting.pop(id_code)
        deleted_value2 = self.informations.pop(id_code)

        return deleted_value1,deleted_value2

    def tik_goes_tak(self):
        for id_code in self.counting:
            self.counting[id_code] = self.counting[id_code] - 1
            if self.counting[id_code] == 0:
                self.remove_temp(id_code)


    def obs_num(self):
        return len(self.informations)

    def add_perm(self,id_code,x_loc,y_loc,color = "b"):
        self.informations[id_code] = [x_loc,y_loc,color]

    def remove_perm(self,id_code):
        deleted_value = self.informations.pop(id_code)
        return deleted_value

    def change_color(self,id_code,color = 'r'):
        self.informations[id_code][2] = color

    def getx(self,id_code):
        return self.informations[id_code][0]

    def gety(self,id_code):
        return self.informations[id_code][1]

    def getcolor(self,id_code):
        return self.informations[id_code][2]

    def update_cord(self,id_code,xx,yy):
        self.informations[id_code] = [xx,yy,self.informations[id_code][2]]

    def update_color(self,id_code,color):
        self.informations[id_code] = [self.informations[id_code][0],self.informations[id_code][1],color]

class Environment:

    def __init__(self,xs = 100,ys = 100):
        ys = xs #!!!!!!!!!!!!!!!!!!!
        self.xs = xs
        self.ys = xs ###!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT CHANGE!
        self.sun = [[1.0]*ys]*xs
        self.instances = [[[] for _ in range(ys)] for _ in range(xs)]
        self.feromons = [[[] for _ in range(ys)] for _ in range(xs)]
        self.spawned_num = 0
        self.avaible_gens = "ABCDEFGHIJKLMNOPQRSTVWXYZ|"
        self.observation = Observation()

    def getsun(self):
        return self.sun

    def getinstances(self):
        return self.instances

    def getferomons(self):
        return self.feromons


    def census_pop_num(self):
        num_identity = {}
        a=1
        for x in range(self.xs):
            for y in range(self.ys):
                if(self.instances[x][y] != []):
                    for ins in self.instances[x][y]:
                        ins.stats.number = a
                        num_identity[ins.stats.get_number()] = ins.stats.get_identity()
                        a= a + 1

        return num_identity


    def Spawn(self, xp=0,yp=0, amount = 1, specie_type = Species()):
        for i in range(1,amount+1):
            specie_type.stats.identity = str(self.spawned_num+i)
            specie_type.cord.xx = xp
            specie_type.cord.yy = yp
            self.instances[xp][yp].append(specie_type)

    def Reproduction(self, xp=0,yp=0, parent_identity = "", num_of_children = 0 , specie_type = Species()):
        specie_type.stats.identity = parent_identity + self.avaible_gens[num_of_children%len(self.avaible_gens)]
        specie_type.fixed.num_of_children = 0
        specie_type.storage.energy = specie_type.evolutionary.reproduction_divide * specie_type.storage.energy
        specie_type.cord.xx = xp
        specie_type.cord.yy = yp
        self.instances[xp][yp].append(specie_type)


    def simulate(self,ticks):
        for tick in range(ticks):
            for x in range(self.xs):
                for y in range(self.ys):
                    if (self.instances[x][y] != []):
                        for ins in self.instances[x][y]:

                            if (ins.storage.energy <= 0):
                                if(ins.stats.ifobserved()):
                                    print("usunieto:",self.observation.remove_temp(ins.stats.get_identity()))
                                self.instances[x][y].remove(ins)
                                break

                            ins.doing(self.sun,self.xs)
                            ins.moving(self.xs)

                            if(ins.stats.ifobserved()):
                                self.observation.update_cord(ins.stats.get_identity(),ins.cord.getx(),ins.cord.gety())


                            if(ins.neural.reproducing == True):
                                self.Reproduction(x, y, ins.stats.identity, ins.fixed.num_of_children, ins)
                                ins.fixed.num_of_children = ins.fixed.num_of_children + 1


    def observe_random(self,color = 'b',ticks = 0):
        for xi in range(self.xs):
            for yi in range(self.ys):
                if self.instances[xi][yi] != []:
                    if (ticks == 0):
                        self.observation.add_perm(self.instances[xi][yi][0].stats.identity,self.instances[xi][yi][0].cord.xx,self.instances[xi][yi][0].cord.yy,color)
                    else:
                        self.observation.add_temp(self.instances[xi][yi][0].stats.identity,self.instances[xi][yi][0].cord.xx,self.instances[xi][yi][0].cord.yy,color,ticks)


    def observe(self,id_code,color = 'b', ticks = 0):
        for xi in range(self.xs):
            for yi in range(self.ys):
                if self.instances[xi][yi] != []:
                    for ins in self.instances[xi][yi]:
                        if id_code == ins.stats.get_identity() :
                            if(ticks == 0):
                                self.observation.add_perm(id_code,ins.stats.xx,ins.stats.yy,color)
                            else:
                                self.observation.add_temp(id_code, ins.stats.xx, ins.stats.yy, color, ticks)

    def ignition_show_cord(self):
        if self.observation.obs_num() == 0 :
            return "(!)Does not observe anybody(!)"
        plt.ion()
        self.fig = plt.figure()



    def show_cord(self):
        if self.observation.obs_num() == 0 :
            return "(!)Does not observe anybody(!)"

        for inf in self.observation.informations.values():


            plt.xlim(0, self.xs)
            plt.ylim(0, self.ys)

            plt.plot([inf[0]], [inf[1]], ".", color = inf[2])

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


        plt.clf()






def dircords(x,y,dir,step,L):

    if(dir == "w"):
        y = y + step
        if(y>L):
            y = L

    if (dir == "a"):
        x = x - step
        if (x < 0):
            x = 0

    if (dir == "s"):
        y = y - step
        if (y < 0):
            y = 0

    if (dir == "d"):
        x = x + step
        if (x > L):
            x = L

    return x,y


def Area_Creator(x,y,rangexy):
    xstart = x - rangexy
    xstop = x + rangexy
    ystart = y - rangexy
    ystop = y + rangexy

    Area = []
    for xs in range(xstart, xstop+1):
        for ys in range(ystart, ystop+1):
            Area.append(0)

    return Area

def Sight_detection_sun(env,x,y,rangexy,MAX_range,L):
    xstart = x - rangexy
    if(xstart<0):
        xstart = 0
    xstop = x + rangexy
    if (xstop > L):
        xstop = L
    ystart = y - rangexy
    if (ystart < 0):
        ystart = 0
    ystop = y + rangexy
    if (ystop > L):
        ystop = L


    operational_Area = Area_Creator(x,y,MAX_range)

    CORD = []
    CORDx = []
    CORDy = []

    xrstep = 2 * MAX_range + 1
    xrstart = 180 - (rangexy)*(xrstep)
    xrstop = 1 + xrstart + (xstop-xstart)*(xrstep)


    yrstop = ystop-ystart+1

    for xs in range(xrstart,xrstop,xrstep):
        for ys in range(yrstop):
            CORD.append(xs+ys)
            CORDx.append(xs)
            CORDy.append(ys)

    n=0
    for cords in CORD:
        operational_Area[cords] = env[CORDx[n]][CORDy[n]]
        n+=1

    return operational_Area







environment = Environment(1000)
environment.Spawn(100,100)
environment.simulate(1)
a = environment.census_pop_num()


environment.ignition_show_cord()
environment.show_cord()
while(True):
    environment.simulate(1)
    environment.show_cord()
