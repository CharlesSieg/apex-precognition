import plotly.express as px
import plotly.graph_objects as go

def create_figure(result):
  fig = go.Figure()
  fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                      mode='lines',
                      name='Train prediction')))
  fig.add_trace(go.Scatter(x=result.index, y=result[1],
                      mode='lines',
                      name='Test prediction'))
  fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                      mode='lines',
                      name='Actual Value')))
  fig.update_layout(
      xaxis=dict(
          showline=True,
          showgrid=True,
          showticklabels=False,
          linecolor='white',
          linewidth=2
      ),
      yaxis=dict(
          title_text='Close (USD)',
          titlefont=dict(
              family='Rockwell',
              size=12,
              color='white',
          ),
          showline=True,
          showgrid=True,
          showticklabels=True,
          linecolor='white',
          linewidth=2,
          ticks='outside',
          tickfont=dict(
              family='Rockwell',
              size=12,
              color='white',
          ),
      ),
      showlegend=True,
      template = 'plotly_dark'
  )

  annotations = []
  annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                xanchor='left', yanchor='bottom',
                                text='Results',
                                font=dict(family='Rockwell',
                                          size=26,
                                          color='white'),
                                showarrow=False))
  fig.update_layout(annotations=annotations)

  return fig
