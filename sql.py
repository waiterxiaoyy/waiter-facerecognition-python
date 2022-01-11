import pymysql
from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet import Facenet
from PIL import Image

db = pymysql.connect(host="127.0.0.1", user="root", password="121223", db="db_school")

cursor = db.cursor()

app = Flask(__name__)
CORS(app, resources=r'/*')

model = Facenet()


@app.route('/url', methods=['POST'])
def func():
    if request.method == "POST":
        username = request.form.get("username")
        nowimage = request.form.get("nowimage")
        cursor.execute("SELECT * from student WHERE username = \"" + str(username) +"\"")
        data = cursor.fetchall()
        temp = {}
        result = []
        info = {}
        info["code"] = 200
        info["msg"] = ""
        if(data!= None):
            for i in data:
                temp["id"] = i[0]
                temp["username"] = i[1]
                temp["userimage"] = i[2]
        try:
            userimage = Image.open(str(temp["userimage"]))
            nowimage = Image.open(nowimage)
        except:
            print('Image Open Error! Try again!')
            return "服务器异常"
        probability = model.detect_image(userimage, nowimage)
        if (probability[0] < 0.9):
            print("Same Sample")
            info["code"] = 200
            info["msg"] = "Same Sample"
            result.append(info.copy())
            return jsonify(result)
        else:
            print("Different Sample")
            info["code"] = 200
            info["msg"] = "Different Sample"
            result.append(info.copy())
            return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8899)
    db.close()
    print("bye")
