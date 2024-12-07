#include <iostream>
#include "IModel.h"
#include "Emotion.h"  // Подключаем модель эмоций
#include "Resnet.h"  // Подключаем другую модель при необходимости

int main() {
    const std::string imageFile = "C:\\Users\\zxc18\\Desktop\\cpp\\emotion\\emotion\\resource\\test.png";
    //const std::string labelFile = "C:\\Users\\zxc18\\Desktop\\cpp\\emotion\\emotion\\resource\\emotion.txt";
    const std::string labelFile = "C:\\cv\\cv2022\\image\\imagenet_classes.txt";
    //const std::wstring modelPath = L"C:\\Users\\zxc18\\Desktop\\cpp\\emotion\\emotion\\resource\\emotion-ferplus-8.onnx";
    const std::wstring modelPath = L"C:\\Users\\zxc18\\Desktop\\SCML\\test__cml_lib\\image\\resnet50v2.onnx";


    IModel* model = new ResnetModel();  // Выбираем нужную модель

    try {
        model->RunModel(imageFile, labelFile, modelPath);  // Вызываем модель
        std::cout << "Model executed successfully!" << std::endl;
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    delete model;  // Освобождаем ресурсы
    return 0;
}
