from flask import Flask,render_template,request
import gui2
app=Flask(__name__,template_folder='templates')
@app.route('/')
def home():
    return render_template('interface.html')

@app.route('/', methods=['POST', 'GET'])
def prediction():
    if request.method=='POST':
        message=request.form.get('massage')
    result=gui2.predict(message)
    
    return render_template('interface.html')

#def predict_api():

if __name__ == "__main__":
    app.run(debug=True)