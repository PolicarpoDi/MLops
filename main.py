from flask import Flask, request, jsonify
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('casas.csv')
colunas = ['tamanho', 'ano', 'garagem']


# variavel explicativa #
x = df.drop('preco', axis=1)
# variavel de resposta #
y = df['preco']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

modelo = LinearRegression()
modelo.fit(x_train, y_train)

app = Flask(__name__)


@app.route('/')
def home():
    return "Minha primeira API"


@app.route('/sentimento/<frase>')
def sentimento(frase):
    tb = TextBlob(frase)
    polaridade = tb.sentiment.polarity
    return f"Polaridade: {polaridade}"


@app.route('/cotacao/', methods=['POST'])
def cotacao():
    # variavel que recebe os dados enviados pelo payload #
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])


app.run(debug=True)