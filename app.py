import pickle
model = pickle.load(open('model.pkl', 'rb'))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np



scaler = pickle.load(open('scaler.pkl', 'rb'))

df = pd.read_csv("AAPL.csv")
df = df.reset_index()['close']


def stock_pred(day):
    lst_output = []
    n_steps = 100
    i=0
    x_input = (scaler.transform(np.array(df[len(df)-100:]).reshape(-1,1))).reshape(1,-1)
    temp_inp = list(x_input)
    temp_inp = temp_inp[0].tolist()

    while(i<day):
        if (len(temp_inp)>100):
            x_input = np.array(temp_inp[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1,n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_inp.extend(yhat[0].tolist())
            temp_inp = temp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1,n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    return scaler.inverse_transform(lst_output)

num = num = st.number_input(
    label="Enter a value",
    min_value=1, 
    step=1,         
    format="%d" 
)

if st.button("Predict"):
    new_lst=[]
    for i in stock_pred(num):
        new_lst.append(i[0])

    fig = go.Figure()
    old_index = np.arange(1,101)
    new_index = np.arange(101, 101+num)
    fig.add_trace(go.Scatter(x=old_index, y=df[len(df)-100:], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=new_index, y=new_lst, mode='lines', name='pred Price', line=dict(color='orange')))
    fig.update_layout(title="AAPL Stock Price", xaxis_title="Date", yaxis_title="Price ($)",legend=dict(x=0, y=1), template="plotly_white")

    st.plotly_chart(fig,use_container_width=True)