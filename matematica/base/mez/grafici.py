from icecream import ic
ic.configureOutput(prefix="> ", includeContext=True)

import plotly.express as px

import numpy as np
import pandas as pd
from plotly.offline import plot
import plotly.graph_objs as go
from . import calcolo, Mat





def grafico_funzione(base_form_lambda, lim, x_disconts, result):
    # ic(x_disconts)
    # ic([str(i) for i in x_disconts])
    specie2_x_discont_nums = [x.as_decimal() for x in x_disconts["specie2"]]
    specie2_x_discont_nums.sort()
    # ic(specie2_x_discont_nums)

    # ic(base_form_lambda)
    lambda_vec = np.vectorize(lambda x: eval(base_form_lambda))
    # ic(lambda_vec)

    if type(lim) == calcolo.Inf:
        lim_num = 0
    else:
        lim_num = lim.as_decimal()


    fig = go.Figure()


    data_points = 1000
    # massimi e minimi della x, +/- 5 di padding per non finire in modo brusco al limite / per avere -15 e +15
    if specie2_x_discont_nums:
        max_lim = max(10, specie2_x_discont_nums[-1]) + 5
        min_lim = min(-10, specie2_x_discont_nums[0]) - 5
    else:
        max_lim = 15
        min_lim = -15

    tot_diff = abs(max_lim) + abs(min_lim)

    if specie2_x_discont_nums:
        ranges = [min_lim] + specie2_x_discont_nums + [max_lim]

    else:
        ranges = [min_lim, max_lim]


    # ic(ranges)
    no_disc_X_ranges = []
    X_ranges = []
    discontinuity_epsilon = 0.001

    for i_r in range(len(ranges))[:-1]:
        # ic(ranges)
        diff = abs(ranges[i_r]) + abs(ranges[i_r+1])

        no_disc_X_ranges.append(np.linspace(ranges[i_r] + 0.15, ranges[i_r+1] - 0.15,
                                    int(data_points*(tot_diff/diff))))
        X_ranges.append(np.linspace(ranges[i_r] + discontinuity_epsilon, ranges[i_r+1] - discontinuity_epsilon,
                                    int(data_points*(tot_diff/diff))))  # equa percentuale di data points


    Y_ranges = [lambda_vec(Xs) for Xs in X_ranges]

    x_axis_range = [lim_num-5, lim_num+5]
    no_disc_X_range = np.concatenate([range_ for range_ in no_disc_X_ranges])
    no_disc_X_range = no_disc_X_range[(no_disc_X_range > x_axis_range[0]) & (no_disc_X_range < x_axis_range[1])]  # seleziona le x più grandi del bordo x iniziale a sx e le x
        # più piccole di quello a dx e poi prende solo gli elementi in comune, quindi le x che rientrano nel range
    Y_no_disc_range = lambda_vec(no_disc_X_range)


    rosso_acceso = "#ff004c"
    rosso_scuro = "#b00e39"
    rosso_porpora = "#8a0c2e"
    trasparente = "rgba(0,0,0,0)"
    plot_bg_color = "rgb(250, 250, 250)"
    grid_color = "rgb(200,200,200)"
    zeroaxis_color = "rgb(220,220,220)"


    # ASINTOTI
    for x_disc in x_disconts["specie2"]:
        x_disc_num = x_disc.as_decimal()

        fig.add_vline(x=x_disc_num, line_dash="dot")
        fig.add_trace(go.Scatter(x=[x_disc_num-0.15, x_disc_num-0.15, x_disc_num+0.15, x_disc_num+0.15], y=[100,-100,-100,100,],  # todo al posto di 100 mettere il max
                                 line_color=trasparente, fillcolor=trasparente, fill="toself",
                                 name=f"Discontinuità di 3° specie in x={x_disc.fancy_str()}", hovertemplate="<extra></extra>"))
                                 # name=f"Discontinuità di 2° specie in x={x_disc.fancy_str()}"))

    # FUNZIONE
    for Xs, Ys in zip(X_ranges, Y_ranges):
        fig.add_trace(go.Scatter(x=Xs, y=Ys, line_shape="spline", line=dict(color=rosso_acceso, width=3),
                                 hovertemplate="x: %{x:.3f} <br> y: %{y:.3f}<extra></extra>"))

    # linea tratteggiata a dx e sx
    left_edge_Xs = np.linspace(min_lim-1, min_lim, num=data_points//tot_diff)
    right_edge_Xs = np.linspace(max_lim, max_lim+1, num=data_points//tot_diff)

    for Xs in [left_edge_Xs, right_edge_Xs]:
        fig.add_trace(go.Scatter(x=Xs, y=lambda_vec(Xs), line_shape="spline", line=dict(color=rosso_acceso, width=3, dash="dash"),
                                 hovertemplate="x: %{x:.3f} <br> y: %{y:.3f}<extra></extra>"))

    # fig.update_traces()

    # DISCONTINUITà 3 SPECIE
    for x_disc, y_val in x_disconts["specie3"]:
        # ic(x_disc)
        # ic(y_val)

        x_disc_num = x_disc.as_decimal()

        # fig.add_trace(go.Scatter(x=[x_disc_num], y=[y_val.as_decimal()], mode="markers",
        fig.add_trace(go.Scatter(x=[x_disc_num], y=[y_val.as_decimal()], mode="markers",
                                 marker=dict(size=12, color=plot_bg_color, line=dict(width=2.5, color=rosso_porpora)),
                                 meta=[f"Discontinuità di 3° specie in x={x_disc.fancy_str()}"], hovertemplate="<b>%{meta[0]}</b>"))

    # LIM MARKER
    # ic(str(type(lim)))
    # ic(str(type(result)))
    if "Inf'>" not in str(type(result)) and "Inf'>" not in str(type(lim)) and result != None:
        y_val = result.as_decimal()
        fig.add_trace(go.Scatter(x=[lim_num], y=[y_val], mode="markers",
                                 marker=dict(size=12, color=rosso_acceso, line=dict(width=2.5, color=rosso_porpora)),
                                 meta=[f"Lim x → {lim.fancy_str()} = {result.fancy_str()}"], name="", hovertemplate="<b>%{meta[0]}</b>"))
    else:
        y_val = 0

    button1 = dict(method = "relayout",
               args = [{"yaxis.scaleanchor": None,
                        "constrain": "domain",
                        "xaxis.range": [x_axis_range[0], x_axis_range[1]],
                        "yaxis.range": [np.min(Y_no_disc_range)-20, np.max(Y_no_disc_range)+20]}],
               label = "␣ y ≠ ␣ x")
    button2 = dict(method = "relayout",
                   args = [{"yaxis.scaleanchor": "x",
                            "yaxis.scaleratio": 1,
                            "yaxis.range": [y_val-5, y_val+5]}],
                   label = "␣ y = ␣ x")

    fig.update_xaxes(
        range=[x_axis_range[0], x_axis_range[1]],
        showspikes=True, spikemode="marker",
        gridwidth=1, gridcolor=grid_color, griddash="dot",
        zerolinewidth=2, zerolinecolor=zeroaxis_color,
        nticks=20
    )
    fig.update_yaxes(
        range=[np.min(Y_no_disc_range)-20, np.max(Y_no_disc_range)+20],
        showspikes=True, spikemode="marker",
        gridwidth=1, gridcolor=grid_color, griddash="dot",
        zerolinewidth=2, zerolinecolor=zeroaxis_color,
        nticks=8
    )

    #Update layout for graph object Figure
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0,r=0,b=0,t=0),
        paper_bgcolor=trasparente,
        plot_bgcolor=plot_bg_color,
        height=400,
        updatemenus=[dict(showactive=True,
                          buttons=[button1, button2], type="buttons",
                          direction="right", pad={"r": 0, "t": 20}, x=0.86, xanchor="left", y=1, yanchor="top")]
    )


    #Turn graph object into local plotly graph
    plot_obj = plot({'data': fig}, output_type='div', config={"modeBarButtonsToRemove": ['lasso2d', 'autoScale2d', 'select2d']})
    return plot_obj

