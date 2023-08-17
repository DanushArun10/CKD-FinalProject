from flask import Flask,render_template,request
import pickle
model = pickle.load(open('CKD.pkl','rb'))
app=Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')
'''
@app.route("/login",methods=['POST'])
def Login():
    return render_template('login.html')
'''
@app.route("/login",methods=['GET', 'POST'])
def Credit():
    '''
    if request.method == "POST":
        Uname=request.form.get("uname")
        Pwd=request.form.get("pwd")'''
    return render_template('predict.html')


@app.route("/result",methods=['GET', 'POST'])
def result():
    p = request.form["sg"]
    q = request.form["hemo"]
    r = request.form["sc"]
    s = request.form["al"]
    t = request.form["pcv"]
    u = request.form["u"]
    if (u == "yes"):
        u1 = 1
    elif (u == "no"):
        u1 = 0
    v = request.form["v"]
    if (v == "yes"):
        v1 = 1
    elif (v == "no"):
        v1 = 0
    x = request.form["bgr"]
    y = request.form["rbc"]
    z = request.form["bu"]

    a = [[float(p), float(q), float(r), float(s), float(t), float(u1), float(v1), float(x), float(y), float(z)]]

    pred = model.predict(a)
    print(pred)
    return render_template('result.html',y=pred)

if __name__=='__main__':
    app.run(debug=True)