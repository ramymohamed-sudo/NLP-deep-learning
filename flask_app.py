

""" Flask and Swagger """
# this file name should be app.py 

from flask import Flask, request
""" Add two number and expose this as API """
app = Flask(__name__)
@app.route('/')     #/ is the default route /predict
def add():
    # a = 10
    a = request.args.get("a")
    # b = 5
    b = request.args.get("b")
    return str(int(a)+int(b))

if __name__ == "__main__":      # __main__ is called hard coded string
    # app.run()
    app.run(host='0.0.0.0',port=7000,debug=True)
    app.run(use_reloader=True)
 
  
