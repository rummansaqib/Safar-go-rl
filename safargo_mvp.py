"""
SafarGo RL MVP — educational synthetic simulator.
Run: pip install streamlit numpy pandas matplotlib
     streamlit run safargo_mvp.py
All data are synthetic; this is not connected to real drivers or SafarGo systems.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SafarGo RL MVP", layout="wide")
st.title("SafarGo — Reinforcement Learning MVP")
st.caption("Synthetic educational simulation only. No real dispatching or GPS data.")
days=st.slider("Simulation duration (days)",1,14,7)
seed=st.number_input("Random seed",min_value=1,max_value=9999,value=23)
zones=40
rng=np.random.default_rng(int(seed))
def demand(t,z):
    h=(t%288)/12
    peak=np.exp(-((h-8)/1.8)**2)+1.1*np.exp(-((h-17.5)/2.2)**2)
    return max(.05,(.35+peak)*(1+.45*np.sin((z+3)*.55)+.2*np.cos(z*.21)))
def run(policy):
    pos=rng.integers(0,zones,300)
    visits=np.zeros(zones); fuel=rides=0; waits=[]; curve=[]; total=0
    for t in range(days*288):
        d=np.array([demand(t,z) for z in range(zones)])
        req=rng.poisson(d*.55); supply=np.bincount(pos,minlength=zones)
        for z in range(zones):
            n=min(int(req[z]),int(supply[z]))
            if n:
                rides+=n; waits.extend([max(1,3+.7*req[z]/max(1,supply[z]))]*n)
                ids=np.where(pos==z)[0][:n]
                pos[ids]=np.clip(z+rng.integers(-7,8,size=len(ids)),0,zones-1)
        ids=rng.choice(300,30,replace=False)
        supply=np.bincount(pos,minlength=zones)
        for i in ids:
            z=int(pos[i])
            if policy=="Random": dest=int(rng.integers(zones))
            elif policy=="Busiest zone": dest=int(np.argmax([demand(t+1,j) for j in range(zones)]))
            else:
                score=np.array([demand(t+1,j)/(1+.12*supply[j])-.025*abs(j-z)+.15/(1+visits[j]/(visits.mean()+1e-6)) for j in range(zones)])
                dest=int(np.argmax(score))
            fuel+=abs(dest-z)*.12; pos[i]=dest; visits[dest]+=1
        w=np.mean(waits[-100:]) if waits else 7
        total += .7*min(rides/(t+1),20)-.18*max(0,w-7)-.08*fuel/(t+1)-.15*np.count_nonzero(visits<max(1,visits.mean()*.25))
        if t%12==0: curve.append(total)
    return {"Policy":policy,"Completed rides":rides,"Average wait (min)":round(float(np.mean(waits)) if waits else 15,2),
            "Fuel proxy (units)":round(float(fuel),2),"Cumulative reward":round(float(total),2),
            "Zones visited":int(np.count_nonzero(visits))},curve
if st.button("Run simulation"):
    rows=[]; curves={}
    for p in ["Random","Busiest zone","RL-inspired"]:
        row,c=run(p); rows.append(row); curves[p]=c
    df=pd.DataFrame(rows)
    st.subheader("Results (synthetic)")
    st.dataframe(df,use_container_width=True,hide_index=True)
    fig,ax=plt.subplots()
    for p,c in curves.items(): ax.plot(np.arange(len(c))*12/288,c,label=p)
    ax.set_xlabel("Days"); ax.set_ylabel("Cumulative reward"); ax.legend()
    st.pyplot(fig)
    st.warning("Toy simulator only. Validate offline and use human-approved shadow mode before any live pilot.")
