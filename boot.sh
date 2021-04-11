#!/bin/sh
exec gunicorn -c main/config/gunicorn.py -b :12500 - app:app
