from flask import Flask, request, jsonify
from lookup import lookup
import json
import base64
import io
app = Flask(__name__)


@app.route('/api/train', methods=['POST'])
def post():
    data = json.loads(request.get_data())
    for annotation in data['annotations']:
        img = base64.b64decode(annotation['document'][2:-1])
        annotation['document'] = io.BytesIO(img)
    result, dimensions = lookup(data['annotations'], data['model_name'])
    if result:
        model = open('trained_models/'+data['model_name']+'.pth', 'rb')
        model_data = str(base64.b64encode(model.read()))
        return jsonify(
            result=result,
            dimensions=dimensions,
            model=model_data
        )
    else:
        return jsonify(
            result=result,
            dimensions=dimensions
        )


@app.route('/')
def main():
    return 'Server to Train Annotator Models.'


if __name__ == '__main__':
    app.run()