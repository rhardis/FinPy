# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:13:37 2019

@author: Richard Hardis
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'hello world'

if __name__ == '__main__':
    app.run()