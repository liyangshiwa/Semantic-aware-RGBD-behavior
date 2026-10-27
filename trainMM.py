
from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM('yolo11n-mm-mid.yaml')
    # model = YOLOMM('yolo11n-mm-mid3.yaml')
    model.train(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
                epochs=200,batch=16,
                # modality='X',
                # cache=True,
                exist_ok=True,
                project='ResTest',name='Test/YOLOMM')
