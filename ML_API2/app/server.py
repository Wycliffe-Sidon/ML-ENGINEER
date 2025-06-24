from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model and preprocessor
model = joblib.load('app/model.joblib')
# Load the preprocessor separately
preprocessor = joblib.load('app/model.joblib') # Corrected line

# Create a FastAPI instance
app = FastAPI()

# Define the input data model using Pydantic
import tkinter as tk
from tkinter import messagebox
from pydantic import BaseModel, ValidationError, Field

# Define the Pydantic model
class StudentData(BaseModel):
    gender: str = Field(..., description="Gender of the student")
    race_ethnicity: str = Field(..., description="Race/Ethnicity of the student")
    parental_level_of_education: str = Field(..., description="Parental education level")
    lunch: str = Field(..., description="Lunch type")
    test_preparation_course: str = Field(..., description="Test preparation status")
    reading_score: int = Field(..., ge=0, le=100, description="Reading score")
    writing_score: int = Field(..., ge=0, le=100, description="Writing score")

# GUI Application
class StudentForm(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student Data Entry")

        # Dictionary to hold label text and Entry widgets
        self.entries = {}

        # Define all fields
        fields = [
            "gender", "race_ethnicity", "parental_level_of_education",
            "lunch", "test_preparation_course", "reading_score", "writing_score"
        ]

        # Create labels and entries dynamically
        for idx, field in enumerate(fields):
            label = tk.Label(self, text=field.replace('_', ' ').title())
            label.grid(row=idx, column=0, pady=5, padx=5, sticky='e')
            entry = tk.Entry(self)
            entry.grid(row=idx, column=1, pady=5, padx=5)
            self.entries[field] = entry

        # Submit button
        btn_submit = tk.Button(self, text="Submit", command=self.submit)
        btn_submit.grid(row=len(fields), column=0, columnspan=2, pady=10)

    def submit(self):
        # Collect data from entries
        data = {}
        for field, entry in self.entries.items():
            val = entry.get()
            if field in ["reading_score", "writing_score"]:
                # Convert to integer where appropriate
                try:
                    val = int(val)
                except ValueError:
                    messagebox.showerror("Invalid input", f"{field.replace('_', ' ').title()} must be an integer")
                    return
            data[field] = val

        # Validate and create instance using Pydantic
        try:
            student = StudentData(**data)
            messagebox.showinfo("Success", f"Student data is valid:\n{student.json(indent=2)}")
        except ValidationError as e:
            messagebox.showerror("Validation Error", str(e))

if __name__ == "__main__":
    app = StudentForm()
    app.mainloop()

# Define a prediction endpoint
@app.post("/predict_math_score/")
async def predict_math_score(data: StudentData):
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Apply the preprocessor to the input data
    processed_input = preprocessor.transform(input_df) # This should now work

    # Now you can make predictions using the loaded model
    prediction = model.predict(processed_input)

    return {"predicted_math_score": prediction.tolist()} # Return the prediction