from flask import Flask, request, render_template
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib # Para salvar e carregar o modelo

app = Flask(__name__)

# --- Treinamento do Modelo (simples para demonstração) ---
# Dados de exemplo: [Vermelho, Amarelo, Rugosidade]
# Saída: 0 para Laranja, 1 para Maçã

# Supondo que Maçã é mais vermelha e menos amarela, e pode ser mais lisa ou um pouco rugosa.
# Laranja é mais amarela, menos vermelha e mais rugosa.
X_train = np.array([
    [0.8, 0.2, 0.3], # Maçã
    [0.7, 0.3, 0.2], # Maçã
    [0.9, 0.1, 0.4], # Maçã
    [0.6, 0.4, 0.35], # Maçã

    [0.2, 0.8, 0.7], # Laranja
    [0.3, 0.7, 0.6], # Laranja
    [0.1, 0.9, 0.8], # Laranja
    [0.4, 0.6, 0.75] # Laranja
])

y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0]) # 1 para Maçã, 0 para Laranja

# Treinando um modelo de Árvore de Decisão
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Salvando o modelo para que não precise ser treinado toda vez
joblib.dump(model, 'fruit_classifier_model.pkl')
print("Modelo treinado e salvo como 'fruit_classifier_model.pkl'")

# --- Carregando o modelo (em um ambiente real, você carregaria em vez de treinar aqui) ---
# model = joblib.load('fruit_classifier_model.pkl')

# --- Rotas do Flask ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Coletando os dados do formulário
            vermelho = float(request.form['vermelho'])
            amarelo = float(request.form['amarelo'])
            rugosidade = float(request.form['rugosidade'])

            # Preparando os dados para o modelo
            features = np.array([[vermelho, amarelo, rugosidade]])

            # Fazendo a previsão
            prediction = model.predict(features)[0]

            # Mapeando o resultado para Maçã ou Laranja
            result = "Maçã" if prediction == 1 else "Laranja"

            return render_template('result.html', result=result, vermelho=vermelho, amarelo=amarelo, rugosidade=rugosidade)
        except ValueError:
            return "Por favor, insira valores numéricos válidos para os nervos."
        except Exception as e:
            return f"Ocorreu um erro: {e}"

if __name__ == '__main__':
    # Cria uma pasta 'templates' se ela não existir
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True) # debug=True reinicia o servidor automaticamente a cada mudança
