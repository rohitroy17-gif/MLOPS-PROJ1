```python
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

# Import project modules
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import (
    VehicleData,
    VehicleDataClassifier,
)
from src.pipline.training_pipeline import TrainPipeline


# ============================================================
# Initialize FastAPI application
# ============================================================

app = FastAPI()


# ============================================================
# Static files
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


# ============================================================
# Jinja2 Templates
# ============================================================

templates = Jinja2Templates(directory="templates")
templates.env.cache = None


# ============================================================
# CORS Configuration
# ============================================================

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DataForm Class
# ============================================================

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    """

    def __init__(self, request: Request):
        self.request: Request = request

        self.Gender: Optional[int] = None
        self.Age: Optional[int] = None
        self.Driving_License: Optional[int] = None
        self.Region_Code: Optional[float] = None
        self.Previously_Insured: Optional[int] = None
        self.Annual_Premium: Optional[float] = None
        self.Policy_Sales_Channel: Optional[float] = None
        self.Vintage: Optional[int] = None
        self.Vehicle_Age_lt_1_Year: Optional[int] = None
        self.Vehicle_Age_gt_2_Years: Optional[int] = None
        self.Vehicle_Damage_Yes: Optional[int] = None

    async def get_vehicle_data(self):
        """
        Retrieve and assign form data to class attributes.
        """

        form = await self.request.form()

        self.Gender = form.get("Gender")
        self.Age = form.get("Age")
        self.Driving_License = form.get("Driving_License")
        self.Region_Code = form.get("Region_Code")
        self.Previously_Insured = form.get("Previously_Insured")
        self.Annual_Premium = form.get("Annual_Premium")
        self.Policy_Sales_Channel = form.get("Policy_Sales_Channel")
        self.Vintage = form.get("Vintage")
        self.Vehicle_Age_lt_1_Year = form.get("Vehicle_Age_lt_1_Year")
        self.Vehicle_Age_gt_2_Years = form.get("Vehicle_Age_gt_2_Years")
        self.Vehicle_Damage_Yes = form.get("Vehicle_Damage_Yes")


# ============================================================
# Home Route
# ============================================================

@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Render the main HTML form page.
    """

    return templates.TemplateResponse(
        request=request,
        name="vehicledata.html",
        context={
            "context": "Rendering"
        },
    )


# ============================================================
# Training Route
# ============================================================

@app.get("/train")
async def trainRouteClient():
    """
    Initiate the model training pipeline.
    """

    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return Response("Training successful!!!")

    except Exception as e:
        return Response(
            f"Error Occurred! {e}"
        )


# ============================================================
# Prediction Route
# ============================================================

@app.post("/")
async def predictRouteClient(request: Request):
    """
    Receive vehicle form data and make a prediction.
    """

    try:

        # Get form data
        form = DataForm(request)
        await form.get_vehicle_data()

        # Create VehicleData object
        vehicle_data = VehicleData(
            Gender=form.Gender,
            Age=form.Age,
            Driving_License=form.Driving_License,
            Region_Code=form.Region_Code,
            Previously_Insured=form.Previously_Insured,
            Annual_Premium=form.Annual_Premium,
            Policy_Sales_Channel=form.Policy_Sales_Channel,
            Vintage=form.Vintage,
            Vehicle_Age_lt_1_Year=form.Vehicle_Age_lt_1_Year,
            Vehicle_Age_gt_2_Years=form.Vehicle_Age_gt_2_Years,
            Vehicle_Damage_Yes=form.Vehicle_Damage_Yes,
        )

        # Convert form data into DataFrame
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        # Initialize prediction pipeline
        model_predictor = VehicleDataClassifier()

        # Make prediction
        value = model_predictor.predict(
            dataframe=vehicle_df
        )[0]

        # Interpret prediction
        status = (
            "Response-Yes"
            if value == 1
            else "Response-No"
        )

        # Render page with prediction result
        return templates.TemplateResponse(
            request=request,
            name="vehicledata.html",
            context={
                "context": status
            },
        )

    except Exception as e:

        return {
            "status": False,
            "error": f"{e}"
        }


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    app_run(
        app,
        host=APP_HOST,
        port=APP_PORT,
    )
```
