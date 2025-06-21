from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import os # Importar para criar a pasta 'templates' e 'static'

app = Flask(__name__)

# --- Treinamento do Modelo (simples para demonstração) ---
# Dados de exemplo: [Vermelho, Amarelo, Rugosidade]
# Saída: 0 para Laranja, 1 para Maçã
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

model = DecisionTreeClassifier(random_state=42) # Adicione random_state para reprodutibilidade
model.fit(X_train, y_train)

# Salvando o modelo
if not os.path.exists('models'): # Criar pasta 'models' se não existir
    os.makedirs('models')
joblib.dump(model, 'models/fruit_classifier_model.pkl')
print("Modelo treinado e salvo como 'models/fruit_classifier_model.pkl'")

# --- Carregando o modelo (em um ambiente real, você carregaria em vez de treinar aqui) ---
# model = joblib.load('models/fruit_classifier_model.pkl')

# --- Rotas do Flask ---

@app.route('/')
def index():
    return render_template('index.html')

# Nova rota para a API de predição que o JavaScript vai chamar
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() # Pega os dados JSON enviados pelo JavaScript
    try:
        vermelho = float(data['vermelho'])
        amarelo = float(data['amarelo'])
        rugosidade = float(data['rugosidade'])

        features = np.array([[vermelho, amarelo, rugosidade]])
        prediction_proba = model.predict_proba(features)[0] # Probabilidade para cada classe
        prediction_class = model.predict(features)[0]

        result = "Maçã" if prediction_class == 1 else "Laranja"

        # Retorna um JSON com as probabilidades e o resultado
        return jsonify({
            'result': result,
            'probabilities': {
                'Laranja': round(prediction_proba[0], 2), # Índice 0 para Laranja (se 0 for Laranja no y_train)
                'Maçã': round(prediction_proba[1], 2)    # Índice 1 para Maçã (se 1 for Maçã no y_train)
            },
            'input_values': {
                'vermelho': vermelho,
                'amarelo': amarelo,
                'rugosidade': rugosidade
            }
        })
    except (ValueError, KeyError) as e:
        return jsonify({'error': f"Dados inválidos: {e}. Certifique-se de enviar 'vermelho', 'amarelo' e 'rugosidade' como números."}), 400
    except Exception as e:
        return jsonify({'error': f"Ocorreu um erro no servidor: {e}"}), 500

# Rota para servir arquivos estáticos (CSS, JS)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    # Cria as pastas 'templates' e 'static' se elas não existirem
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('models'):
        os.makedirs('models')

    app.run(debug=True)
