import numpy as np
import math, sys, string
sys.path.append('lib')
sys.path.append('../../../MaterialsLibrary')
from matplotlib import pyplot as plt
from matplotlib import rc
plt.rcParams['text.usetex'] = True
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['font.family'] = ['Times New Roman']
rc('text', usetex=True)
rc('font', family='serif')
import pathlib
from read_data import *

#Font Sizes
textfont = 32 #Use for x axis and y axis labels
axisfont = 30 #Use for anottations, xticks and yticks, legends
legendfont = 20 #Gets in the way if any bigger

#DataFolders
datafolder = str(pathlib.Path().resolve()) #current directory path
matfilepath = datafolder + '/Zr_MAT.txt'; #directory of material file

#Experimental Data
EXPIa=np.array([[0.1245,1.3249],[0.3034,1.1033],[0.6301,1.163],[0.8557,1.0831],
    [1.3613,1.4861],[1.5946,1.3451],[1.7191,1.4055],[2.5514,1.3048],[2.7381,1.2645],
    [3.7105,1.6474],[4.4417,2.0101],[4.6283,2.5945],[5.7174,2.9572],
    [6.0596,2.9773],[6.1141,3.1184],[6.6197,3.4207],[7.2887,4.3073],[8.1677,5.1537]])  #IODIDE PURITY a # x dpa y strain
EXPIc=np.array([[ 0.1400,-0.1461],[0.3034,-0.4081],[0.6379, -0.4081],[1.2368,-0.7103],
    [1.5013,-0.8111],[1.6180,-0.6902],[2.1936,-0.6902],[2.6059,-1.3149],
    [3.1037,-1.1537],[3.5860,-1.2544],[4.2083,-1.0126]]) #IODIDE PURITY c # x dpa y strain
EXPZa=np.array([[0.1245,0.5189],[0.2723,0.7406],[0.4745,0.8816],[0.6845,1.1234],[1.0346,1.4660],[1.2757,1.5063],
    [1.5013,1.2645],[1.6024,1.0428],[1.9758,1.4660],[2.3492,1.2645],[3.1271,1.7884],
    [3.3293,1.8690],[3.6482,2.1310],[3.9594,2.1713],[4.4261,2.3325],[4.6283,2.1713],
    [5.6940,2.4332],[5.9507,2.9370], [6.4719,2.8766],[6.6430,3.1385],[6.7753,3.1788]])  #Zone Refined a # x dpa y strain units e-4
EXPZc=np.array([[0.1478,-0.4282],[0.2800,-0.2267],[0.6145,-0.7909],[0.8634,-0.7506],[1.3613,-0.7506],
    [1.4935,-0.6297],[1.8124,-0.7708],[2.1625,-0.8917],[2.3647,-0.7103],[3.4693,-1.0126]]) #Zone Refined c # x dpa y strain units e-4

#Read in diffusive, plastic, volumetric strains
F,Flabels=readFfile(datafolder)
betaD11 = getFarray(F,Flabels,'betaD11 [-]') ; betaD12 = getFarray(F,Flabels,'betaD12 [-]') ;
betaD13 = getFarray(F,Flabels,'betaD13 [-]') ; betaD22 = getFarray(F,Flabels,'betaD22 [-]') ;
betaD23 = getFarray(F,Flabels,'betaD23 [-]') ; betaD33 = getFarray(F,Flabels,'betaD33 [-]') ;
CbetaP11 = getFarray(F,Flabels,'CbetaP11 [-]') ; CbetaP12 = getFarray(F,Flabels,'CbetaP12 [-]') ;
CbetaP13 = getFarray(F,Flabels,'CbetaP13 [-]') ; CbetaP22 = getFarray(F,Flabels,'CbetaP22 [-]') ;
CbetaP23 = getFarray(F,Flabels,'CbetaP23 [-]') ; CbetaP33 = getFarray(F,Flabels,'CbetaP33 [-]') ;
A1betaP11 = getFarray(F,Flabels,'A1betaP11 [-]') ; A1betaP12 = getFarray(F,Flabels,'A1betaP12 [-]') ;
A1betaP13 = getFarray(F,Flabels,'A1betaP13 [-]') ; A1betaP22 = getFarray(F,Flabels,'A1betaP22 [-]') ;
A1betaP23 = getFarray(F,Flabels,'A1betaP23 [-]') ; A1betaP33 = getFarray(F,Flabels,'A1betaP33 [-]') ;
A2betaP11 = getFarray(F,Flabels,'A2betaP11 [-]') ; A2betaP12 = getFarray(F,Flabels,'A2betaP12 [-]') ;
A2betaP13 = getFarray(F,Flabels,'A2betaP13 [-]') ; A2betaP22 = getFarray(F,Flabels,'A2betaP22 [-]') ;
A2betaP23 = getFarray(F,Flabels,'A2betaP23 [-]') ; A2betaP33 = getFarray(F,Flabels,'A2betaP33 [-]') ;
A3betaP11 = getFarray(F,Flabels,'A3betaP11 [-]') ; A3betaP12 = getFarray(F,Flabels,'A3betaP12 [-]') ;
A3betaP13 = getFarray(F,Flabels,'A3betaP13 [-]') ; A3betaP22 = getFarray(F,Flabels,'A3betaP22 [-]') ;
A3betaP23 = getFarray(F,Flabels,'A3betaP23 [-]') ; A3betaP33 = getFarray(F,Flabels,'A3betaP33 [-]') ;
CbetaV11 = getFarray(F,Flabels,'CbetaV11 [-]') ; CbetaV12 = getFarray(F,Flabels,'CbetaV12 [-]') ;
CbetaV13 = getFarray(F,Flabels,'CbetaV13 [-]') ; CbetaV22 = getFarray(F,Flabels,'CbetaV22 [-]') ;
CbetaV23 = getFarray(F,Flabels,'CbetaV23 [-]') ; CbetaV33 = getFarray(F,Flabels,'CbetaV33 [-]') ;
A1betaV11 = getFarray(F,Flabels,'A1betaV11 [-]') ; A1betaV12 = getFarray(F,Flabels,'A1betaV12 [-]') ;
A1betaV13 = getFarray(F,Flabels,'A1betaV13 [-]') ; A1betaV22 = getFarray(F,Flabels,'A1betaV22 [-]') ;
A1betaV23 = getFarray(F,Flabels,'A1betaV23 [-]') ; A1betaV33 = getFarray(F,Flabels,'A1betaV33 [-]') ;
A2betaV11 = getFarray(F,Flabels,'A2betaV11 [-]') ; A2betaV12 = getFarray(F,Flabels,'A2betaV12 [-]') ;
A2betaV13 = getFarray(F,Flabels,'A2betaV13 [-]') ; A2betaV22 = getFarray(F,Flabels,'A2betaV22 [-]') ;
A2betaV23 = getFarray(F,Flabels,'A2betaV23 [-]') ; A2betaV33 = getFarray(F,Flabels,'A2betaV33 [-]') ;
A3betaV11 = getFarray(F,Flabels,'A3betaV11 [-]') ; A3betaV12 = getFarray(F,Flabels,'A3betaV12 [-]') ;
A3betaV13 = getFarray(F,Flabels,'A3betaV13 [-]') ; A3betaV22 = getFarray(F,Flabels,'A3betaV22 [-]') ;
A3betaV23 = getFarray(F,Flabels,'A3betaV23 [-]') ; A3betaV33 = getFarray(F,Flabels,'A3betaV33 [-]') ;
MobilebetaV11 = getFarray(F,Flabels,'MobilebetaV11 [-]') ; MobilebetaV12 = getFarray(F,Flabels,'MobilebetaV12 [-]') ;
MobilebetaV13 = getFarray(F,Flabels,'MobilebetaV13 [-]') ; MobilebetaV22 = getFarray(F,Flabels,'MobilebetaV22 [-]') ;
MobilebetaV23 = getFarray(F,Flabels,'MobilebetaV23 [-]') ; MobilebetaV33 = getFarray(F,Flabels,'MobilebetaV33 [-]') ;
#loops only
AbetaCD11=A1betaP11+A2betaP11+A3betaP11+A1betaV11+A2betaV11+A3betaV11;
AbetaCD22=A1betaP22+A2betaP22+A3betaP22+A1betaV22+A2betaV22+A3betaV22;
AbetaCD33=A1betaP33+A2betaP33+A3betaP33+A1betaV33+A2betaV33+A3betaV33;
CbetaCD11=CbetaP11+CbetaV11;
CbetaCD22=CbetaP22+CbetaV22;
CbetaCD33=CbetaP33+CbetaV33;
#total strain
betaCD11=betaD11+AbetaCD11+CbetaCD11+MobilebetaV11;
betaCD22=betaD22+AbetaCD22+CbetaCD22+MobilebetaV22;
betaCD33=betaD33+AbetaCD33+CbetaCD33+MobilebetaV33;

#Get dpa
TimeStep = get_scalar(matfilepath,'CDtimeStep')
DoseRate = get_scalar(matfilepath,'G0_SI') #dose rate
print("DDTimestep =",TimeStep,"Dose Rate =",DoseRate)
s = 0.5; dpa = []; #starting dpa ~0.6
l = len(betaCD11);
for x in range(l):
    dpa.append(s+x*(DoseRate*TimeStep)) #Dose*TimeStep = dpa0 + runID*(dpa/runID)

dpa2 = dpa;
#append starting zero
dpa = np.append(0,dpa)
betaCD11 = np.append(0,betaCD11)
betaCD22 = np.append(0,betaCD22)
betaCD33 = np.append(0,betaCD33)
print(dpa)

#plot data
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.15,top=0.95,left=0.15,right = 0.95, wspace=0.05)

#experimental
ax.scatter(EXPIa[:,0],EXPIa[:,1]*10**-2,color="firebrick",marker="s",s=80,label=r'Experiment: Iodide Purity');
ax.scatter(EXPZa[:,0],EXPZa[:,1]*10**-2,color="orange",marker="s",s=80,label=r'Experiment: Zone Refined Purity');
ax.scatter(EXPIc[:,0],EXPIc[:,1]*10**-2,color="firebrick",marker="s",s=80);
ax.scatter(EXPZc[:,0],EXPZc[:,1]*10**-2,color="orange",marker="s",s=80);

#simulation
ax.plot(dpa,betaCD11*100,'k',linewidth=2,label=r'Simulation: Total'); #growth %
ax.plot(dpa,betaCD22*100,'k',linewidth=2); #growth %
ax.plot(dpa,betaCD33*100,'k',linewidth=2); #growth %
ax.plot(dpa2,AbetaCD11*100,'r-.',linewidth=2,label=r'Simulation: Interstitial Loops');
ax.plot(dpa2,AbetaCD22*100,'r-.',linewidth=2);
ax.plot(dpa2,AbetaCD33*100,'r--',linewidth=2);
ax.plot(dpa2,CbetaCD11*100,'b-.',linewidth=2,label=r'Simulation: Vacancy Clusters $\&$ Loops');
ax.plot(dpa2,CbetaCD22*100,'b-.',linewidth=2);
ax.plot(dpa2,CbetaCD33*100,'b--',linewidth=1);

# Annotate
ax.annotate(r'$\langle a \rangle$ axis',fontsize=axisfont, xy=(8, .02), xytext=(8, .02));
ax.annotate(r'$\langle c \rangle$ axis', fontsize=axisfont, xy=(8, -.02), xytext=(8, -.025));

# Plot editing 
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.75);
ax.set_xlim(0)
ax.set(xlabel=r'Irradiation Dose (dpa)', ylabel=r'Growth Strain ($\%$)');
ax.legend(loc='lower left',fontsize=legendfont);
plt.yticks(np.arange(-0.08, 0.07, 0.02))
plt.xticks(np.arange(0, 10, 1.0))

ax.tick_params(axis = 'both', which = 'major', direction='in', bottom=True, left=True, top=True, right=True, labelsize = axisfont)
ax.tick_params(axis = 'both', which = 'minor', direction='in', bottom=True, left=True, top=True, right=True, labelsize = axisfont)
for item in ([ax.xaxis.label, ax.yaxis.label]): item.set_fontsize(textfont)

##################################################
# Save Figures
##################################################
#plt.savefig(datafolder+'/../../Figures/Figure3a.pdf', format="pdf", dpi=1200)
#plt.savefig(datafolder+'/../../GraphicalAbstract/Growth.pdf', format="pdf", dpi=1200)
plt.show()

#  #Calculating Growth Without Swelling
#  a1strain =np.array([]); ystrain =np.array([]); cstrain =np.array([]);
#  for k in range(0,len(swell)):
#    #Inelastic Matrix:
#    Inelastic[0,0] = betaCD11[k]; Inelastic[1,1] = betaCD22[k]; Inelastic[2,2] = betaCD33[k];
#    Inelastic[0,1] = betaCD12[k]; Inelastic[1,0] = betaCD12[k];
#    Inelastic[0,2] = betaCD13[k]; Inelastic[2,0] = betaCD13[k];
#    Inelastic[1,2] = betaCD23[k]; Inelastic[2,1] = betaCD23[k];
#    #Swelling Matrix:
#    swelling = I*swell[k]
#    # Calculate Growth:
#    Gstrain = Inelastic - swelling
#    # Append a1, a2, a3, c for plotting
#    print("Swelling",swelling)
#    a1strain = np.append(a1strain,np.array(Gstrain[0,0]))
#    ystrain = np.append(ystrain,np.array(Gstrain[1,1]))
#    cstrain = np.append(cstrain,np.array(Gstrain[2,2]))
#    #Print Out Growth Matrix
#    print("Growth Strain Tensor for simulation #" + str(x) + " at " + str(dpa[k]) + " dpa \n", Gstrain)
#    # Plot a1 and c axis growth strain
#    colorx = ["dodgerblue","red","green","orange"]
#  #ax.plot(dpa,a1strain,color=colorx[x], label='Great White Simulation' + str(x));
#  #ax.plot(dpa,cstrain,color=colorx[x]);
