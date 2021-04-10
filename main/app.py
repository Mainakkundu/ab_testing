import flask
import logging
from flask import jsonify, request
from main.utils import status

#define base app
app=flask.Flask(__name__)
current_version='v0'

@app.route(f'/{current_version}/base',methods=['get'])
def base():
    response=jsonify({'msg':'AB testing code running successfully!!'})
    response.status_code=status.HTTP_200_OK
    return(response)


if __name__=='__main__':
    app.run(host='0.0.0.0',port='5000')

if __name__!='__main__':
    app.logger.setLevel(logging.INFO)
