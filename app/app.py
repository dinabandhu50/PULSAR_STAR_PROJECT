from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict_func():
    data = request.get_json()
    pred = model.predict(data)
    return jsonify(str(pred))


if __name__ == '__main__':
    with open('./models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    app.run(debug=True)
