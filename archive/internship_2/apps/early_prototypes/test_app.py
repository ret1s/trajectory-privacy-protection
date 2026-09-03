from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <html>
        <body>
            <h1>Flask is working!</h1>
            <p>If you see this, Flask is running correctly.</p>
            <p><a href="/test">Test another route</a></p>
        </body>
    </html>
    '''

@app.route('/test')
def test():
    return '<h1>Test route works too!</h1>'

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='127.0.0.1')