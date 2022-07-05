import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def Adenosine_Phosphates(CHEP_System, ATP, VO2a, VO2max, Pi, Lam, Sa=7, Sc=25, M3=0.96):
    # calculating hydrogen ion concentration
    pCO2 = 40 + 55 * (VO2a / VO2max)
    dbuff = 1 / 54
    pHm = 7.85 + dbuff * (0.8 * Pi - Lam) - 0.55 * np.log10(pCO2)
    H = 10 ** (-pHm)

    # Calculating Phosphates
    PCr = CHEP_System - ATP
    M1 = H * 1.66 * 10 ** 9
    Q = M1 * (PCr / (Sc - PCr))
    ADP = Sa * Q / (M3 + Q + Q ** 2)
    ATP = ADP * Q
    AMP = ADP ** 2
    Pi = Sc - PCr

    return PCr, Pi, ATP, ADP, AMP, M1, H, pHm

# Model with ODEs also considering Glycogen depletion (VO2max + VLamax)
def Model(t, y, ATP_Belastung, VO2max, vlamax, Ks1, Ks2, Ks3, Kelox, KpyrO2, T,
          VolRel_m, VolRel_b, Sc, Sa, M3, ATP_Ruhe, Vrel, PCr, Pi, ATP, ADP, AMP, M1, H, kdiff):

    # Allocating state of parameters to values within function
    GP, VO2a, Lam, Lab = y

    ## System of ODEs
    # 1. ODE
    dGP = VO2a * 0.233 + ((1 / (1 + (H ** 3 / Ks3))) * (vlamax / (1 + (Ks2 / (ADP * AMP))))) * 1.4 - ATP_Belastung - ATP_Ruhe

    # 2. ODE
    dVO2a = ((VO2max / (1 + (Ks1 / (ADP ** 2)))) - VO2a) / T

    # 3. ODE
    dLam = (VolRel_m ** -1) * (((1 / (1 + (H ** 3 / Ks3))) * (vlamax / (1 + (Ks2 / (ADP * AMP))))) - ((KpyrO2 * VO2a) / (1 + ((VolRel_m ** 2 * Kelox) / Lam ** 2)))) - (kdiff * (Lam - Lab))

    # 4. ODE
    dLab = Vrel * (kdiff * (Lam - Lab) - ((KpyrO2 * VO2a) / (1 + ((Vrel ** 2 * Kelox) / (Lab ** 2)))))

    return np.array([dGP, dVO2a, dLam, dLab])


## Call_Mader, function needed to initiate the calculations
# It takes 20 positional arguments, all have default values
# ATP_demand = demand in ATP-Equivalents per kg muscle per second; y02 = array containing initial parameters ([CHEP, VO2a, Lam, Lab]),
# VO2max = VO2max in ml/min/kgmuscle, vlamax = mmol/min/kgmuscle, Muscle_Active = Active compartment in relation to overall (!) body as decimal,
# Ks1 = halfmaximal activation of OxP 0.035 ** 2 (must be to power of two!), Ks2 = halfmaximal activation of glycolysis 0.15 ** 3 (must be to power of 3), Ks3 = halfmaximal activation of pH inhibition 10 ** -20.2, Kelox = halfmaximal activation of PDH (no power),
# ...other constants according to book


def call_Mader(ATP_demand=np.repeat(250, 30*60), y0s=np.array([25, 350 / 60 / 18.2, 1.1, 1.1]),
               VO2max=150, vlamax= 60, Muscle_Aktiv=0.26,
               Ks1=0.035 ** 2, Ks2=0.15 ** 3, Ks3=10 ** -20.2, Kelox=2, KpyrO2=0.01475, T=10, VolRel_m=0.75,
               VolRel_b=0.45, Sc=25, Sa=7, M3=0.96, ATP_temp=5, Pi_temp=5, H_temp=10 ** -7, ATP_Ruhe=0, Glycogen_full=25, VO2max_depleted_perc = 0.7):

    ## Allocating Input to variables within the function
    ATP = ATP_temp
    H = H_temp
    Pi = Pi_temp
    CHEP_System = np.array([y0s[0]])
    OxP = np.array([y0s[1]])
    Lactate_Muscle = np.array([y0s[2]])
    Lactate_Blood = np.array([y0s[3]])
    ATP_store = np.array([ATP_temp])
    pHm_store = np.array([np.log10(H_temp) * -1])
    AMP_store = np.array([0.0001])
    ADP_store = np.array([0.0001])
    Glycogen_store = np.array([Glycogen_full])
    Glycogen_perc = Glycogen_store / Glycogen_full
    VO2max_glyc_perc = VO2max_depleted_perc + (1 - VO2max_depleted_perc) * Glycogen_perc ** (1 / 4)

    # Controlling time span for RK45
    t_span = np.array([0, 1])

    # converting units
    vlamax = vlamax / 60
    VO2max = VO2max / 60
    Vrel = Muscle_Aktiv / (VolRel_b - Muscle_Aktiv)

    # for loop solving system of differential equations for every 1 second interval
    for n in range(0, len(ATP_demand)):

        # using initial values in first loop
        if n == 0:
            PCr, Pi, ATP, ADP, AMP, M1, H, pHm = Adenosine_Phosphates(CHEP_System, ATP, y0s[1], VO2max, Pi, y0s[2], Sa=Sa, Sc=Sc, M3=M3)
            kdiff = 0.065 * (y0s[3] ** -1.4)

        # using previously calculated values from second loop to end
        else:
            PCr, Pi, ATP, ADP, AMP, M1, H, pHm = Adenosine_Phosphates(x.y[0][1], ATP, x.y[1][1], VO2max, Pi, x.y[2][1], Sa=Sa, Sc=Sc, M3=M3)
            kdiff = 0.065 * (x.y[3][1] ** -1.4)

        # RK4 solving differential equations
        x = solve_ivp(Model, t_span, y0s, args=(
            ATP_demand[n],VO2max_glyc_perc * VO2max, Glycogen_perc * vlamax, Ks1, Ks2, Ks3, Kelox, KpyrO2, T,
            VolRel_m, VolRel_b, Sc, Sa, M3, ATP_Ruhe, Vrel, PCr, Pi, ATP, ADP, AMP, M1, H, kdiff), first_step=1, method="RK45", dense_output=True, vectorized=True)

        # storing data
        CHEP_System = np.append(CHEP_System, x.y[0][1])
        OxP = np.append(OxP, x.y[1][1])
        Lactate_Muscle = np.append(Lactate_Muscle, x.y[2][1])
        Lactate_Blood = np.append(Lactate_Blood, x.y[3][1])
        ATP_store = np.append(ATP_store, ATP)
        pHm_store = np.append(pHm_store, pHm)
        AMP_store = np.append(AMP_store, AMP)
        ADP_store = np.append(ADP_store, ADP)
        Glycogen_store = np.append(Glycogen_store, (Glycogen_store[-1] - (((1 / (1 + (H ** 3 / Ks3))) * (vlamax / (1 + (Ks2 / (ADP * AMP))))) / 1000 / 2 * 180.156)))


        # Setting new start values for next loop
        y0s = np.array([x.y[0][1], x.y[1][1], x.y[2][1], x.y[3][1]])
        Glycogen_perc = Glycogen_store[-1]/Glycogen_full
        VO2max_glyc_perc = VO2max_depleted_perc + (1 - VO2max_depleted_perc) * Glycogen_perc ** (1/4)


        # Contractioninsufficiency Criteria PCr <= 1
        if x.y[0][1] - ATP <= 1:
            break
        elif Glycogen_store[-1] < 0.1:
            break

    # reporting ending status
    if x.message == 'The solver successfully reached the end of the integration interval.':
        print('Calculation successfull')
    else:
        print('Beware. An Error occured during the calculation!')

    return CHEP_System, OxP, Lactate_Muscle, Lactate_Blood, ATP_store, pHm_store, AMP_store, ADP_store, Glycogen_store

# Creating an athlete for the simulation
VO2max = 150
VLamax = 60
Muscle_Aktiv = 0.26
VolRel_b = 0.45
bw = 70

# Creating a bout of exercise, here a step test
Steps = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]
ATP_demand = np.array([])
for Step in Steps:
    ATP_demand = np.append(ATP_demand, np.repeat(Step, 5*60) / (bw * Muscle_Aktiv) * 0.0465)

# calling the function
CHEP_System, OxP, Lactate_Muscle, Lactate_Blood, ATP_store, pHm_store, AMP_store, ADP_store, Glycogen_store = call_Mader(ATP_demand= ATP_demand, VO2max=VO2max, vlamax=VLamax, VolRel_b=VolRel_b)

# Creating Visualisations
## Lactate kinetics in blood [mmol/L] and muscle [mmol/kgm]
time_min = np.arange(0, len(Lactate_Blood), 1) / 60
plt.plot(time_min, Lactate_Blood, label = 'Blood')
plt.plot(time_min, Lactate_Muscle, label = 'Muscle')
plt.xlabel('Time [min]')
plt.ylabel('Lactate Concentration')
plt.legend()
plt.show()

## PCr Stores [mmol/kgm]
plt.plot(time_min, CHEP_System-ATP_store, label = 'PCr Store')
plt.xlabel('Time [min]')
plt.ylabel('[mmol/kgm]')
plt.legend()
plt.show()

## Oxygen utilization (ml / min / kg)
plt.plot(time_min, OxP * 60 * Muscle_Aktiv, label= 'Oxygen Utilization')
plt.xlabel('Time [min]')
plt.ylabel('ml / min / kg')
plt.legend()
plt.show()

## Glycogen Store
plt.plot(time_min, Glycogen_store, label= 'Glycogen Store')
plt.xlabel('Time [min]')
plt.ylabel('[mmol/kgm]')
plt.legend()
plt.show()

