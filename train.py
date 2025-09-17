from ultralytics import YOLO
import torch



def main():
    data = 'datasets/new_sliced/dataset.yaml'
    # data = 'medical-pills.yaml'


    device = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 400
    batch = 32

    save_period = 100
    cache = True

    imgsz=640

    # data augmentation settings
    fliplr = 0.5
    shear = 0
    scale = 0.3     # the ball changes its size when moving away from camera, so the I find scaleing benefitial
    bgr = 0.1
    mosaic = 1.0    # recommended by YOLO for better detection of smaller objects
    degrees = 20

    classes = [0]   # only train for class 'ball'

    model_name = 'yolo11n'
    project = 'gala_sliced'

    print(device)
    model = YOLO(model_name)
    model = model.train(
        data=data,
        device=device,
        epochs=epochs,
        batch=batch,
        save_period=save_period,
        classes=classes,
        cache=cache,
        fliplr=fliplr,
        shear=shear,
        scale=scale,
        imgsz=imgsz,
        bgr=bgr,
        degrees=degrees,
        mosaic=mosaic,
        project=project
    )


if __name__ == '__main__':
    main()
