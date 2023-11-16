import plotly.express as px
import plotly.graph_objects as go

# def create_figure(result):
#   fig = go.Figure()
#   fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
#                       mode='lines',
#                       name='Train prediction')))
#   fig.add_trace(go.Scatter(x=result.index, y=result[1],
#                       mode='lines',
#                       name='Test prediction'))
#   fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
#                       mode='lines',
#                       name='Actual Value')))
#   fig.update_layout(
#       xaxis=dict(
#           showline=True,
#           showgrid=True,
#           showticklabels=False,
#           linecolor='white',
#           linewidth=2
#       ),
#       yaxis=dict(
#           title_text='Close (USD)',
#           titlefont=dict(
#               family='Rockwell',
#               size=12,
#               color='white',
#           ),
#           showline=True,
#           showgrid=True,
#           showticklabels=True,
#           linecolor='white',
#           linewidth=2,
#           ticks='outside',
#           tickfont=dict(
#               family='Rockwell',
#               size=12,
#               color='white',
#           ),
#       ),
#       showlegend=True,
#       template = 'plotly_dark'
#   )

#   annotations = []
#   annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
#                                 xanchor='left', yanchor='bottom',
#                                 text='Results',
#                                 font=dict(family='Rockwell',
#                                           size=26,
#                                           color='white'),
#                                 showarrow=False))
#   fig.update_layout(annotations=annotations)

#   return fig

plot_height = 1000
plot_width = 1300
plot_template = "plotly_dark"

def plot_forecast(forecast):
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=forecast.index, y=forecast.actual, mode='lines', name='actual'))
  fig.add_trace(go.Scatter(x=forecast.index, y=forecast.forecast, mode='lines', name='forecast'))
  fig.update_layout(
    autosize=False, height=plot_height, template=plot_template, width=plot_width
  )
  fig.show()

def plot_history(data):
  fig = px.line(data["Close"])
  fig.update_layout(
    autosize=False, height=plot_height, template=plot_template, width=plot_width,
    xaxis=dict(
      linecolor='white',
      linewidth=2,
      showgrid=True,
      showline=True,
      showticklabels=False,
      title_text='Close (USD)',
      titlefont=dict(
        family='Rockwell',
        size=12,
        color='white',
      ),
    ),
  )
  # annotations = []
  # annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
  #                               xanchor='left', yanchor='bottom',
  #                               text='Results',
  #                               font=dict(family='Rockwell',
  #                                         size=26,
  #                                         color='white'),
  #                               showarrow=False))
  fig.show()

def plot_losses(tr, va):
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=tr, y=va, mode='lines', name='train'))
  fig.update_layout(
    autosize=False, height=plot_height, template=plot_template, width=plot_width
  )
  fig.show()

def plot_predictions(pdf):
  #fig = go.Figure()
  fig = px.line(pdf)
  fig.update_layout(autosize=False, width=2400, height=800,)
  fig.show()
