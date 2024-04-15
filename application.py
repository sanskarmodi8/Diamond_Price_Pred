from flask import Flask,request,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from flask_cors import CORS

application=Flask(__name__)

app=application

CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return jsonify({'message': 'Diamond price prediction app backend'})
    
    else:
        try:
            data = CustomData(
                carat=float(request.json.get('carat')),
                depth=float(request.json.get('depth')),
                table=float(request.json.get('table')),
                x=float(request.json.get('x')),
                y=float(request.json.get('y')),
                z=float(request.json.get('z')),
                cut=request.json.get('cut'),
                color=request.json.get('color'),
                clarity=request.json.get('clarity')
            )
            final_new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)
            results = str(round(pred[0], 2))

            return jsonify({'price': results})

        except Exception as e:
            return jsonify({'error': str(e)}), 500  # Internal Server Error

    

if __name__=="__main__":
    app.run(host='0.0.0.0',port=50)
