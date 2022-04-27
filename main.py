# import libararies
import uvicorn ##ASGI
from fastapi import FastAPI, File, UploadFile, Form

from ml_components import predict_nail_type, predict_nail_shape, confirm_hear_disease
from api_components import read_imagefile


# 2. Create the app object
app = FastAPI()

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post("/predict/nail_type")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "PNG")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict_nail_type(image)
    return {"prediction":prediction}

@app.post("/predict/nail_shape")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "PNG")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict_nail_shape(image)
    return {"prediction":prediction}

@app.post("/confirm/diabetes")
async def confirm_bedibts(pregnancies: int = Form(...), glucose: int = Form(...),bloodPressure: int = Form(...),insulin: int = Form(...),bmi: float = Form(...),age: int = Form(...)):
    print(pregnancies, glucose,bloodPressure,insulin,bmi,age)
    result = confirm_hear_disease(pregnancies, glucose,bloodPressure,insulin,bmi,age)
    print(result[0])
    return {"prediction": False if result[0] == 0 else True}


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn main:app --reload