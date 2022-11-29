from flask import Flask,render_template,request
import gui2
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('interface.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    message=request.form['massage']
    result=gui2.predict(message)
    
    return render_template('interface.html',prediction_text="{}".format(result))

#def predict_api():

if __name__ == "__main__":
    app.run(port=5000,debug=True)