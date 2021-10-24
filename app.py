import pickle
from flask import Flask, request, render_template, url_for
app = Flask(__name__)


model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('scale.pkl', 'rb'))

@app.route('/',methods=['GET', 'POST'])
def homepage():


    return render_template('index.html')


@app.route('/result',methods=['GET', 'POST'])
def operation():
    if request.method=='POST':
        req=request.form
        SepalLength=float(req.get('s_length'))
        SepalWidth=float(req.get('s_width'))
        PetalLength=float(req.get('p_length'))
        PetalWidth=float(req.get('p_width'))
        x = [[SepalLength, SepalWidth,PetalLength, PetalWidth]]

        y_user_predict = model.predict(x)
        if y_user_predict[0]=='Iris-setosa':

            return render_template('setosa.html')
        elif y_user_predict[0]=='Iris-versicolor':

            return render_template('versicolor.html')

        elif y_user_predict[0]=='Iris-virginica':

            return render_template('virginica.html')

if __name__ == "__main__":
    app.run(debug=True)



