from flask import Flask

app = Flask(__name__)

@app.route("/")
def homepage():
    return """
    <h1>歡迎來到 aivtuber</h1>
    <p>這是首頁</p>
    <p><a href="/xxx/">前往功能頁（/xxx/）</a></p>
    """

@app.route("/xxx/")
def xxx_page():
    return """
    <h1>這是功能頁 /xxx/</h1>
    <p>這裡有你想要的特殊功能</p>
    <a href="/">回首頁</a>
    """

if __name__ == "__main__":
    app.run()
