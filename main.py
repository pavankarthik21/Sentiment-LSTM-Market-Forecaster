from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from GS_predict import predict1
from JP_predict import predict2

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

class Item(BaseModel):
    type: str

@app.post("/gs/")
def gs(item: Item):
    print(item)
    if(item.type=="button1"):
        try:
            fig = plt.figure()
            #plot sth
            # tmpfile = BytesIO()
            
            # binary_fc       = open("images\\image1.png", 'rb').read()  # fc aka file_content
            # base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
            # ext     = filepath.split('.')[-1]
            # dataurl = f'data:image/{ext};base64,{base64_utf8_str}'
            # fig.savefig(dataurl, format='png')
            # encoded = base64.b64encode(dataurl.getvalue()).decode('utf-8')
            
            a=predict1()
            print(a)
            return a
            # print(predict())
            # html = 'Some html head' + '<img src='data:image/png;base64,{}'>'.format(encoded) + 'Some more html'
            # return ({a[1].value,a[2].value})
            # return {1:"hi"}
        except Exception as error:
        # handle the exception
            print("An exception occurred:", error)
    elif(item.type=="button2"):
        try:
            fig = plt.figure()
            #plot sth
            tmpfile = BytesIO()
            fig.savefig(tmpfile, format='png')
            # encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            a=predict2()
            print(a)
            return(a)
            # print(predict())
            # html = 'Some html head' + '<img src='data:image/png;base64,{}'>'.format(encoded) + 'Some more html'
            # return ({a[1].value,a[2].value})
            # return {1:"hi"}
        except Exception as error:
        # handle the exception
            print("An exception occurred:", error)
    
    
