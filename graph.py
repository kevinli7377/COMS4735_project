import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

def graph1():
    x = [100,150,200,300,400,450]
    y = [0,1.0294204871048818,6.1397425793180416,12.131034129362384,15.612987631548325, 17.51962259947324]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title("Difference between Query and GroundTruth Volume Estimation For 100 ml Query", fontsize=16)
    plt.xlabel("Reference Image Volume (ml)", fontsize=14)
    plt.ylabel("Difference (%)", fontsize=14)
    plt.grid(True)

    plt.show()

def graph2():
    x = [30,50,100,150,250,300,350,400,450]
    y = [80.40318719552126,32.415322499968435,9.190958168720376,8.073740918138195,0,-2.6355052804520773,-4.114730241008768,-5.565727648225105,-7.098465915174834]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title("Difference between Query and GroundTruth Volume Estimation For Diferent Queries with Fixed 250ml Reference", fontsize=16)
    plt.xlabel("Query True Volume (ml)", fontsize=14)
    plt.ylabel("Estimation Difference (%)", fontsize=14)
    plt.grid(True)

    plt.show()

def graph3():
    x = [100,150,200,300,400,450]
    y = [-14.927946248429735,-14.042441715273546,-9.694492306307045,-4.596999020383863,-1.634493448583052,0]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title("Difference between Query and GroundTruth Volume Estimation For 450ml Query with Different References", fontsize=16)
    plt.xlabel("Reference Image Volume (ml)", fontsize=14)
    plt.ylabel("Estimation Difference (%)", fontsize=14)
    plt.grid(True)

    plt.show()
# graph2()

graph3()