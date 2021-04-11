from flask import Flask,request, url_for, redirect, render_template, jsonify
from main.utils import status
current_version='v0'

app = Flask(__name__)

@app.route('/')
def home():
    response=jsonify({'msg':'AB home page running successfully!!'})
    response.status_code=status.HTTP_200_OK
    return(response)

@app.route(f'/{current_version}/base',methods=['get'])
def base():
    response=jsonify({'msg':'AB subpage code running successfully!!'})
    response.status_code=status.HTTP_200_OK
    return(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
