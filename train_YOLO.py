from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="data_set/data.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("data_set/test/images/512_1_mp4-45_jpg.rf.e68fd218826a9c245be6c50e36dfb413.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format