from flask import Flask, render_template, send_file, make_response
from corona import table_world,table_india, total_world, total_india,plot_world,plot_tests, plot_severity, plot_india, plot_daily, plot_recovery,plot_age,summary_india,summary_world
from flask import *
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)


@app.route('/')
def show_tables():
    table_w = table_world()
    table_i=table_india()
    sw=summary_world()
    si=summary_india()
    figureDisplay1 = plot_world()
    figureDisplay2=plot_india()
    plot_d=plot_daily()
    plot_a=plot_age()
    plot_r=plot_recovery()
    totalw=total_world()
    totali=total_india()
    plott=plot_tests()
    plot_s=plot_severity()
    return render_template('index.html', tableworld=table_w,tableindia = table_i, total_world=totalw,total_india=totali,
                    figure1=figureDisplay1, plottests=plott,plotseverity=plot_s, figure2=figureDisplay2, 
                    plotd=plot_d, plotr=plot_r,plota=plot_a,sumw=sw,sumi=si)
                    
if __name__ == "__main__":
    app.jinja_env.cache = {}
    app.run(debug=True)