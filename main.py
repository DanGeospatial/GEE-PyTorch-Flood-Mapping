"""


"""
# Set this to TRUE if the flood perimeter model has already been trained
trained = False
# Identifier for model version
model_version = 1


def CollectData():
    pass


def InferenceFloodPerimeter():
    pass


def TrainFloodPerimeter():
    pass


def BuildDataset(model_version: int):
    pass


def TrainModel(model_version: int):
    pass


def AnalyzeResults(model_version: int):
    pass


if __name__ == '__main__':
    # Run analysis
    print("Starting Analysis...")
    CollectData()
    print("Data Collected!")
    if trained:
        print("Using Pre-Trained Flood Perimeter Model")
        InferenceFloodPerimeter()
        print("Flood Perimeter Inferenced!")
    else:
        print("Training Flood Perimeter Model")
        TrainFloodPerimeter()
        InferenceFloodPerimeter()
        print("Flood Perimeter Inferenced!")
    BuildDataset(model_version)
    print("Dataset Created!")
    TrainModel(model_version)
    print("Model Training Complete!")
    AnalyzeResults(model_version)
    print("Model Diagnostics Complete!")
    print("Analysis Finished.")