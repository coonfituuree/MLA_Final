"""Local FastAPI service. Synchronous inference runs in Starlette's thread pool."""
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from backend.schemas import VacancyInput, ScreeningResponse, ModelName
from shared.inference import Predictor
from shared.review import REVIEW_VERSION, POLICY_VERSION, POLICY

MAX_FILE_BYTES = 120000
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    app.state.predictor = None
    app.state.load_error = None
    try:
        app.state.predictor = await run_in_threadpool(Predictor)
    except Exception as exc:
        logger.exception('Unable to load model bundle')
        app.state.load_error = f'{type(exc).__name__}: {exc}'
    yield
    app.state.predictor = None

app = FastAPI(title='VacancyLens', version='1.0.0', lifespan=lifespan)

from backend.limits import RequestSizeLimit
app.add_middleware(RequestSizeLimit)

def get_predictor(request):
    if request.app.state.predictor is None:
        raise HTTPException(503, 'Model bundle unavailable. Run python scripts/download_data.py and python -m training.train from the project virtual environment, then restart the backend. ' + (request.app.state.load_error or ''))
    return request.app.state.predictor

@app.get('/api/health')
def health():
    return {'status':'ok'}

@app.get('/api/ready')
def ready(request: Request):
    predictor = get_predictor(request)
    return {'status':'ready', 'model_version':predictor.config['model_version'], 'device':predictor.device}

@app.get('/api/v1/model-info')
def model_info(request: Request):
    predictor = get_predictor(request)
    return {**predictor.config, 'metrics':predictor.metrics, 'review_layer':{'version':REVIEW_VERSION,'method':'heuristic','policy_version':POLICY_VERSION,'policy':POLICY}, 'available_models':[m for m in predictor.config['thresholds'] if m!='majority_baseline']}

@app.post('/api/v1/predictions', response_model=ScreeningResponse)
def predict(payload: VacancyInput, request: Request):
    try:
        return get_predictor(request).predict(payload.model_dump(), payload.model)
    except ValueError as exc:
        raise HTTPException(422,str(exc))

@app.post('/api/v1/predictions/file', response_model=ScreeningResponse)
async def predict_file(request: Request, file: UploadFile = File(...), model: ModelName = Form('neural_network'), later_instructions: str = Form('',max_length=20000)):
    predictor = get_predictor(request)
    try:
        if not file.filename or not file.filename.lower().endswith('.txt'):
            raise HTTPException(422, 'Upload a UTF-8 .txt file.')
        raw = await file.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise HTTPException(413, 'Text file exceeds 120,000 bytes.')
        try:
            text = raw.decode('utf-8-sig', errors='strict')
        except UnicodeDecodeError:
            raise HTTPException(422, 'Text file must use UTF-8 encoding.')
        if '\x00' in text:
            raise HTTPException(422, 'Binary or NUL-containing files are not supported.')
        try:
            payload = VacancyInput(description=text, model=model, later_instructions=later_instructions)
        except ValidationError as exc:
            raise HTTPException(422, '; '.join(error['msg'] for error in exc.errors()))
        try:
            return await run_in_threadpool(predictor.predict, payload.model_dump(), model)
        except ValueError as exc:
            raise HTTPException(422,str(exc))
    finally:
        await file.close()
