from fastapi import FastAPI , Form , UploadFile, File
from pydantic import BaseModel
import  json


app = FastAPI()

class Item (BaseModel):
    test_acc: float
    test_loss: float
class body(BaseModel):
    epochs : int
    batches : int
    optimizer : str
    loss_function :str
    traning_perc  :float
    dataset_folder : str


@app.get("/")
def read_root():
    return{ "test_acc":"100",
            "loss_acc":"0"}
@app.get("/get_output/")
async def item(item: body):

    return item
