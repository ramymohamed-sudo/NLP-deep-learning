

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
 

from flask import Flask, request
import ktrain 
predictor = ktrain.load_predictor('*.h5') 
def get_pred(x):
    sent = predictor.predict(x)
    return sent[0]
app = Flask(__name__)
@app.route('/',methods='POST')     #/ is the default route /predict
def get_sent():
    tx = request.get_json(force=True)
    text = tx['Review']
    sent = get_pred(text)
    return sent

if __name__ == "__main__":      # __main__ is called hard coded string
    # app.run()
    app.run(host='0.0.0.0',port=7000,debug=True)
    app.run(use_reloader=True)
    



    
    
# JSON JavaScript Object Notation:
>> import requests, json 
# load(), loads(), dump(), dumps()
>> import json
>> json.loads(str(data_dict))		return dictionary
>> data = {‘review’: “This is a great movie. I loved it.”}
>> data = json.dumps(data)			# compare with jsonify(result=sent)
>> url = ‘htt[://…..’
>> x = requests.post(url,data=data)
>> json.dumps(data_str)		return string
To write in json file
>> file = open(“data.json”,’w’)
>> json.dump(data_str,”data.json”)
>> file.close 
To read from  json file
>> file = open(“data.json”,’w’)
>> json_data = json.load(file)
>> file.close 
>>  json.loads(json_data)		to give a return dictionary
