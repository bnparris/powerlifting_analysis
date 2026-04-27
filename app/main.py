from fastapi import FastAPI, HTTPException
from src.predict import predict
from pydantic import BaseModel


app = FastAPI( title='Powerlifting Churn API',
    description='Predicts whether a lifter will compete the following calendar year'
)

class PredictionRequest(BaseModel):
    names: list[str]

#not strictly needed as constructing API response myslef but kept so it shows up in docs
class PredictionResponse(BaseModel):
    predictions: dict[str, int]
    last_complete_year: int
    predicting_for_year: int
    name_not_found: list[str]


@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict', response_model=PredictionResponse)
def predict_churn(request: PredictionRequest):
    if not request.names:
        raise HTTPException(status_code=400, detail='No names provided')

    results = predict(request.names)

    #if no names are found then raise error, but if some are found predict for those that are found
    if len(request.names) == len(results['name_not_found']):
        print('NO MATCHING NAMES')
        raise HTTPException(status_code = 404, detail = 'No matching names were found in the database')
    return {'predictions': results['predictions'],
            'name_not_found': results['name_not_found'],
           'last_complete_year': results['last_complete_year'],
           'predicting_for_year': results['last_complete_year']+1,
           }