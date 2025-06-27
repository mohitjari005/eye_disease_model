import base64
import uvicorn
from fastapi import FastAPI,Request
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, UploadFile,Form
from fastapi.responses import HTMLResponse
from eye_disease_repo.main import predictEyeDisease

app = FastAPI()

templates = Jinja2Templates(directory = "templates")

# @app.get("/")
# async def root():
#     return {"message": "Hello ml "} 
 
@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    base64_data = base64.b64encode(contents).decode("utf-8")
    mime_type = file.content_type  # e.g., "image/jpeg"
    data_url = f"data:{mime_type};base64,{base64_data}"
    print(data_url)

    class_name = predictEyeDisease(data_url)

    return templates.TemplateResponse("upload_form.html", {
        "request": request,
        "uploaded": True,
        "filename": file.filename,
        "image_data": data_url,
        "predicted_class" : class_name

    })

@app.get("/Image", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("Image.html", {"request": request, "image_base64": None})


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8081)


