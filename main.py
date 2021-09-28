from flask import Flask
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('casas.csv')
colunas = ['tamanho', 'preco']
df = df[colunas]

x = df.drop('preco', axis=1)
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


@app.route('/cotacao/<int:tamanho>')
def cotacao(tamanho):
    preco = modelo.predict([[tamanho]])
    return str(preco)


app.run(debug=True)